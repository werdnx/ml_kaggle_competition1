import os
import sys

import librosa
import numpy as np
from tqdm import tqdm

from config import TRAIN_PATH, SAMPLE_RATE, PREPROCESS_PATH


def main():
    data_folder = sys.argv[1]
    # df = pd.read_csv(os.path.join(data_folder, 'train_ground_truth.csv'), dtype={0: str, 1: str})
    files = [(os.path.join(TRAIN_PATH, i), i) for i in os.listdir(TRAIN_PATH)]
    print("Audio preprocessing")
    for file in tqdm(files):
        # row = df.iloc[ind]
        # path = os.path.join(TRAIN_PATH, row[0])
        # path = path + '.wav'
        sound, r = librosa.load(file[0], sr=SAMPLE_RATE, mono=True)
        sound = librosa.util.normalize(sound, axis=0)
        name = file[1].split(".")[0]
        np.save(PREPROCESS_PATH + name + '.npy', sound)


if __name__ == "__main__":
    main()
