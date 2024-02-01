import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QualityClassifier(nn.Module):
    def __init__(self, args, stop_threshold, cache=None):
        super(QualityClassifier, self).__init__()
        self.n_timesteps = args.n_timesteps
        self.stop_threshold = stop_threshold
        if cache is None:
            from vllm import LLM, SamplingParams
            # self.llm = LLM(model='mistralai/Mistral-7B-Instruct-v0.1')
            self.llm = LLM(model='lmsys/vicuna-13b-v1.5-16k')
            self.sampling_params = SamplingParams(temperature=0.8, top_p=0.95, logprobs=100)
            # self.possible_answers_tokens = [28741, 28760, 28743, 28757]  # for mistral: A, B, C, D
            self.possible_answers_tokens = [29909, 29933, 29907, 29928]  # for vicuna: A, B, C, D
            self.cache = None
        else:
            self.cache = cache

    def forward(self, X):
        with torch.no_grad():
            return self._forward(X)

    def _forward(self, X):
        assert type(X) is list
        if self.cache is None:
            all_samples_answer_probs = []
            for i in range(len(X)):
                print(f'Processing sample {i+1}/{len(X)}')
                sample_answer_probs = []
                answers = X[i]['answers']
                print(f'A. {answers[0]}\nB. {answers[1]}\nC. {answers[2]}\nD. {answers[3]}')
                for t in range(self.n_timesteps):
                    context = '. '.join(X[i]['parts'][:t+1])
                    question = X[i]['question']
                    answers = X[i]['answers']

                    prompt = f'''I will give you a question, 4 possible answers (A, B, C, D), and a context. Your task is to choose the correct answer. You must output only the letter of the correct answer\n The context is:\n{context}.\nThe question is: {question}\nThe possible answers are:\nA. {answers[0]}\nB. {answers[1]}\nC. {answers[2]}\nD. {answers[3]}'''
                    # prompt_template=f'''<s>[INST] {prompt} [/INST]\n\nThe answer is: \n\n'''
                    prompt_template = f'''A chat between a curious user and an artificial intelligence assistant. The assistant gives the correct answer to the question of the user, which can be A, B, C or D. USER: {prompt} ASSISTANT:\n\nThe answer is:\n\n'''
                    logprobs = self.llm.generate(prompt_template, self.sampling_params)[0].outputs[0].logprobs[0]
                    A_prob = np.exp(logprobs[self.possible_answers_tokens[0]]) if self.possible_answers_tokens[0] in logprobs else 0
                    B_prob = np.exp(logprobs[self.possible_answers_tokens[1]]) if self.possible_answers_tokens[1] in logprobs else 0
                    C_prob = np.exp(logprobs[self.possible_answers_tokens[2]]) if self.possible_answers_tokens[2] in logprobs else 0
                    D_prob = np.exp(logprobs[self.possible_answers_tokens[3]]) if self.possible_answers_tokens[3] in logprobs else 0
                    answers_prob = torch.tensor([A_prob, B_prob, C_prob, D_prob]).cuda()

                    # Scale the probabilities so they sum to 1
                    answers_prob = answers_prob / answers_prob.sum()
                    if True:
                        answers_prob = answers_prob / answers_prob.sum()
                        print(f'guess={answers_prob.argmax().item()} ({int(answers_prob[answers_prob.argmax().item()]*100)}%) correct={X[i]["label"]} ({int(answers_prob[X[i]["label"]]*100)}%)')
                    sample_answer_probs.append(answers_prob)
                all_samples_answer_probs.append(torch.stack(sample_answer_probs, dim=0))
            all_samples_answer_probs = torch.stack(all_samples_answer_probs, dim=0)

            all_scores = all_samples_answer_probs  # (batch_size, n_timesteps, num_classes)
        else:
            all_scores = torch.stack([self.cache[sample['id']]['all_scores'] for sample in X], dim=0)
        # Take the softmax probability of most likely answer as the heuristic for the probability of being correct
        all_is_correct_estimation = all_scores.max(dim=-1).values  # (batch_size, n_timesteps)

        all_scores, all_is_correct_estimation = all_scores.cpu(), all_is_correct_estimation.cpu()

        if isinstance(self.stop_threshold, torch.Tensor):
            stop_threshold = self.stop_threshold.cpu()
        else:
            stop_threshold = torch.tensor(self.stop_threshold)
        # From here the code is the same as LSTMClassifier
        should_stop = all_is_correct_estimation >= stop_threshold
        should_stop[:, -1] = True  # Always stop at the last time step
        halt_timesteps = should_stop.float().argmax(dim=-1)
        
        # Get scores at halt timesteps
        scores = all_scores[torch.arange(all_scores.shape[0]), halt_timesteps]
        is_correct_estimation = all_is_correct_estimation[torch.arange(all_is_correct_estimation.shape[0]), halt_timesteps]
        
        return {
            'scores': scores,
            'halt_timesteps': halt_timesteps,
            'is_correct_estimation': is_correct_estimation,
            'all_scores': all_scores,
            'all_is_correct_estimation': all_is_correct_estimation,
        }

    def predict_proba(self, x):
        scores = self.forward(torch.from_numpy(x))['scores']
        return F.softmax(scores, dim=-1).detach().cpu().numpy()
