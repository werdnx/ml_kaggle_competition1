import torch
from torch.utils.data import Dataset

# A,B,C,D,E,F,G,H,I

CATEGORIES = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8}


class FeatureDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        return torch.tensor(self.X[index]).float(), self.Y[index]

    def __len__(self):
        return len(self.Y)
