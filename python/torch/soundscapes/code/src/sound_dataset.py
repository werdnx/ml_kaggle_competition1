import os

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import torch

# A,B,C,D,E,F,G,H,I
from audioutils import get_samples_from_file
from config import PREPROCESS_PATH

CATEGORIES = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8}


class SoundDataset(Dataset):
    def __init__(self, base, df, model_param):
        self.base = base
        self.model_param = model_param
        self.data = []
        self.labels = []
        self.length = 0
        for ind in tqdm(range(len(df))):
            row = df.iloc[ind]
            crops = process_npy_file(PREPROCESS_PATH, row[0], model_param['SECONDS'])
            for crop in crops:
                self.length = self.length + 1
                self.data.append(crop[np.newaxis, ...])
                self.labels.append(CATEGORIES[row[1]])

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.length


class SoundDatasetValidation(Dataset):
    def __init__(self, base, df):
        self.df = df
        self.base = base
        self.labels = []
        self.data = []
        for ind in tqdm(range(len(df))):
            row = df.iloc[ind]
            self.labels.append(CATEGORIES[row[1]])
            crops = process_npy_file(PREPROCESS_PATH, row[0])
            self.data.append(torch.from_numpy(crops))

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.df)


def process_npy_file(path, name, seconds):
    path = os.path.join(path, name)
    path = path + '.npy'
    crops = get_samples_from_file(path, seconds)
    # crops = get_one_sample_from_file(path)
    return crops


def sampler_label_callback(dataset, index):
    return dataset.labels[index]
