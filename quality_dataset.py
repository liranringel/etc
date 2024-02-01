import json

from torch.utils.data import Dataset


class QualityDataset(Dataset):
    QUESTION_KEYS = ('question_unique_id', 'question', 'options', 'gold_label')

    def __init__(self, args):
        dataset = [json.loads(j) for j in open('./quality/QuALITY.v1.0.1.htmlstripped.train').read().split('\n') if len(j) > 0]
        dataset += [json.loads(j) for j in open('./quality/QuALITY.v1.0.1.htmlstripped.dev').read().split('\n') if len(j) > 0]
        dataset += [json.loads(j) for j in open('./quality/QuALITY.v1.0.1.htmlstripped.test').read().split('\n') if len(j) > 0]
        self.n_timesteps = 10
        self.num_classes = 4
        X = []
        y = []
        for i in range(len(dataset)):
            sample = dataset[i]
            context = sample['article']
            sentences = context.split('. ')
            if len(sentences) < self.n_timesteps:
                continue
            num_sentences_per_timestep = len(sentences) // self.n_timesteps
            parts = []
            for t in range(self.n_timesteps - 1):
                start = t * num_sentences_per_timestep
                end = (t+1) * num_sentences_per_timestep
                parts.append('. '.join(sentences[start:end]))
            start = (self.n_timesteps - 1) * num_sentences_per_timestep
            parts.append('. '.join(sentences[start:]))
            for question in dataset[i]['questions']:
                if not all(k in question for k in self.QUESTION_KEYS):
                    continue
                X.append({
                    'id': question['question_unique_id'],
                    'parts': parts,
                    'question': question['question'],
                    'answers': question['options'],
                    'label': question['gold_label'] - 1,  # 1-indexed to 0-indexed
                })
                y.append(question['gold_label'] - 1)  # 1-indexed to 0-indexed
        self.X = X
        self.y = y


    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)
