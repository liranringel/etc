import torch
from torch.utils.data import Dataset
from aeon.datasets import load_classification

class TscDataset(Dataset):
    def __init__(self, dataset_name):
        print(f'Loading dataset {dataset_name}...')
        X, y, meta_data = load_classification(dataset_name, return_metadata=True)
        self._class_values = meta_data['class_values']
        y = torch.tensor([self._class_values.index(c) for c in y.tolist()]).long()
        X = torch.from_numpy(X).float()
        assert(len(X.shape)) == 3
        X = X.transpose(2, 1)  # (n_samples, n_features, n_timesteps)
        self.X = X
        self.y = y
        self.n_timesteps = X.shape[1]
        self.input_size = X.shape[2]
        self.num_classes = len(self._class_values)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)
