import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMClassifier(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, num_classes, stop_threshold):
        super(LSTMClassifier, self).__init__()
        self.stop_threshold = stop_threshold
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_scores = nn.Linear(hidden_size, num_classes)
        self.fc_is_correct = nn.Linear(hidden_size, 1)

    def forward(self, x):
        if type(x) is list:
            x = torch.stack(x, dim=0)

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=x.device)

        # Initialize cell state with zeros
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        all_scores = self.fc_scores(out)  # (batch_size, n_timesteps, num_classes)
        all_is_correct_estimation = F.sigmoid(self.fc_is_correct(out)).squeeze(-1)  # (batch_size, n_timesteps)

        should_stop = all_is_correct_estimation >= self.stop_threshold
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
