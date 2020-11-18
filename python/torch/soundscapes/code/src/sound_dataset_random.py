import os

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

# A,B,C,D,E,F,G,H,I
from audioutils import get_random_sample_from_file, get_samples_from_file, samples_in_file
from config import PREPROCESS_PATH, AUGMENT

CATEGORIES = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8}


class SoundDatasetRandom(Dataset):
    def __init__(self, base, df, model_params):
        self.df = df
        self.model_params = model_params
        self.base = base
        self.labels = []
        self.length = 0
        self.index_map = {}
        global_idx = 0
        for ind in tqdm(range(len(df))):
            row = df.iloc[ind]
            samples = sapmles_in_f(PREPROCESS_PATH, row[0], model_params['SECONDS'])
            for i in range(samples):
                self.index_map[global_idx] = ind
                global_idx += 1
                self.labels.append(CATEGORIES[row[1]])
            self.length += samples

    def __getitem__(self, index):
        # format the file path and load the file
        row = self.df.iloc[self.index_map[index]]
        # read from preprocessed npy file
        sound_formatted = process_npy_file(PREPROCESS_PATH, row[0], self.model_params['SECONDS'])
        return sound_formatted[np.newaxis, ...], self.labels[index]

    def __len__(self):
        return self.length


def process_npy_file(path, name, seconds):
    path = os.path.join(path, name)
    path = path + '.npy'
    crops = get_random_sample_from_file(path, seconds, aug=AUGMENT)
    # crops = get_one_sample_from_file(path)
    return crops


def process_npy_file_all(path, name, seconds):
    path = os.path.join(path, name)
    path = path + '.npy'
    crops = get_samples_from_file(path, seconds)
    # crops = get_one_sample_from_file(path)
    return crops


def sapmles_in_f(path, name, seconds):
    path = os.path.join(path, name)
    path = path + '.npy'
    return samples_in_file(path, seconds)


def sampler_label_callback(dataset, index):
    return dataset.labels[index]
