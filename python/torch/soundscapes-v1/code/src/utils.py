import os

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from audioutils import spec_to_image, get_melspectrogram_db

# A,B,C,D,E,F,G,H,I
CATEGORIES = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8}


class ESC50Data(Dataset):
    def __init__(self, base, df, in_col, out_col):
        self.df = df
        self.data = []
        self.labels = []
        # self.c2i = {}
        # self.i2c = {}
        # self.categories = sorted(df[out_col].unique())
        # for i, category in enumerate(self.categories):
        #     self.c2i[category] = i
        #     self.i2c[i] = category
        for ind in tqdm(range(len(df))):
            row = df.iloc[ind]
            file_path = os.path.join(base, row[in_col])
            file_path = file_path + '.wav'
            self.data.append(spec_to_image(get_melspectrogram_db(file_path))[np.newaxis, ...])
            self.labels.append(CATEGORIES[row[out_col]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
