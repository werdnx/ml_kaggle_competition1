import os
import warnings

import librosa
import noisereduce as nr
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from numpy import save
from scipy.ndimage import maximum_filter1d
from sklearn.utils import shuffle
from tqdm import tqdm
from uuid import uuid4

sns.set()
TRAIN_DIR = '/input/'
OUT_DIR = '/input/sample_slides/'
DF = '/input/sample_slides/samples_df'
EPOCHS = 10
BATCH_SIZE = 32
SAMPLES = 256

## Пройтись по каждому файлу
## нарезать файл по 5 сек
samples_from_file = []


def split_waves(audio_file_path, class_label):
    wave_data, wave_rate = librosa.load(audio_file_path)
    wave_data, _ = librosa.effects.trim(wave_data)
    # only take 5s samples and add them to the dataframe
    sample_length = 5 * wave_rate
    for idx in range(0, len(wave_data), sample_length):
        song_sample = wave_data[idx:idx + sample_length]
        if len(song_sample) >= sample_length:
            data = [(song_sample, wave_rate)]
            filename = str(uuid4()) + ".npy"
            path = "{}{}".format(OUT_DIR, filename)
            np.save(path, data)
            samples_from_file.append({"song_sample": path "bird": class_label})


def main():
    train_df = pd.read_csv(TRAIN_DIR + 'train.csv')
    train_df.head()
    print('unique birds ' + str(len(train_df.ebird_code.unique())))
    train_df = shuffle(train_df)
    train_df.head()
    print('train_df len ' + str(len(train_df)))

    features = []
    with tqdm(total=len(train_df)) as pbar:
        for idx, row in train_df.iterrows():
            pbar.update(1)
            try:
                audio_file_path = TRAIN_DIR + 'train_audio/'
                audio_file_path += row.ebird_code
                class_label = row["ebird_code"]
                audio_file_path += row.filename
                split_waves(audio_file_path, class_label)
            except ZeroDivisionError:
                print("{} is corrupted".format(audio_file_path))
    samples_df = pd.DataFrame(samples_from_file)
    samples_df.to_pickle(DF)


if __name__ == "__main__":
    main()
