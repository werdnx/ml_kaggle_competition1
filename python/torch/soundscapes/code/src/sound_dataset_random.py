import os

from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np

# A,B,C,D,E,F,G,H,I
from audioutils import get_random_sample_from_file
from config import PREPROCESS_PATH

CATEGORIES = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8}


class SoundDatasetRandom(Dataset):
    def __init__(self, base, df):
        self.df = df
        self.base = base
        self.labels = []
        for ind in tqdm(range(len(df))):
            row = df.iloc[ind]
            self.labels.append(CATEGORIES[row[1]])

    def __getitem__(self, index):
        # format the file path and load the file
        row = self.df.iloc[index]
        # read from preprocessed npy file
        sound_formatted = process_npy_file(PREPROCESS_PATH, row[0])

        return sound_formatted[np.newaxis, ...], self.labels[index]

    def __len__(self):
        return len(self.df)


def process_npy_file(path, name):
    path = os.path.join(path, name)
    path = path + '.npy'
    crops = get_random_sample_from_file(path)
    # crops = get_one_sample_from_file(path)
    return crops


def sampler_label_callback(dataset, index):
    return dataset.labels[index]
