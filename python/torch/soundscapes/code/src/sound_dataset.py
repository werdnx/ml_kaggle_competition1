import os

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

# A,B,C,D,E,F,G,H,I
from config import AUGMENT, PREPROCESS_PATH, DEF_FREQ
from utils import process_sound

CATEGORIES = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8}
# Frequency of samples, more frequent for less data labels
# TODO calibrate reqs
LABELS_TO_FREQ = {0: 5, 1: 5, 2: 5, 3: 5, 4: 5, 5: 5, 6: 5, 7: 5, 8: 5}


class SoundDataset(Dataset):
    # rapper for the UrbanSound8K dataset
    # Argument List
    #  path to the UrbanSound8K csv file
    #  path to the UrbanSound8K audio files
    #  list of folders to use in the dataset

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
        path = os.path.join(PREPROCESS_PATH, row[0])
        path = path + '.npy'
        preprocessed = np.load(path)
        sound_formatted = process_sound(preprocessed, DEF_FREQ, AUGMENT)

        return sound_formatted, self.labels[index]

    def __len__(self):
        return len(self.df)


def sampler_label_callback(dataset, index):
    return dataset.labels[index]