import os
from uuid import uuid4

import cv2
import librosa
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.utils import shuffle
from tqdm import tqdm

sns.set()
TRAIN_DIR = '/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1/input/bird/'
OUT_DIR = '/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1/input/bird/mel/'


def mono_to_color(
        X: np.ndarray, mean=None, std=None,
        norm_max=None, norm_min=None, eps=1e-6
):
    # Stack X as [X,X,X]
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V


IMG_SIZE = 224
SAMPLES = 512
samples_from_file = []


def extract_features_mfcc_mel(file_path, folder_path, name, class_label):
    if os.path.exists(folder_path + '/' + name + '_mfccs_m.npy') or not os.path.exists(file_path):
        return
    else:
        restored = np.load(file_path, allow_pickle=True)
        wave_data = restored[0][0]
        wave_rate = restored[0][1]
        sample_length = 5 * wave_rate
        i = 0
        correlation_arr = []
        for idx in range(0, len(wave_data), sample_length):
            song_sample = wave_data[idx:idx + sample_length]
            if len(song_sample) >= sample_length:
                correlation_arr.append([])
                mel = librosa.feature.melspectrogram(song_sample, n_mels=SAMPLES)
                db = librosa.power_to_db(mel).astype(np.float32)
                image = mono_to_color(db)
                height, width, _ = image.shape
                image = cv2.resize(image, (int(width * IMG_SIZE / height), IMG_SIZE))
                image = np.moveaxis(image, 2, 0)
                image = (image / 255.0).astype(np.float32)
                filename = str(uuid4()) + ".tif"
                cv2.imwrite(OUT_DIR + filename, image)
                samples_from_file.append({"song_sample": "{}{}".format(OUT_DIR, filename), "bird": class_label})


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
                parts = row.filename.split(".")
                class_label = row["ebird_code"]
                if os.path.exists(audio_file_path + '/' + parts[0] + '.npy'):
                    extract_features_mfcc_mel('{}/{}.npy'.format(audio_file_path, parts[0]), audio_file_path, parts[0],
                                              class_label)
            except ZeroDivisionError:
                print("{} is corrupted".format(audio_file_path))


if __name__ == "__main__":
    main()
