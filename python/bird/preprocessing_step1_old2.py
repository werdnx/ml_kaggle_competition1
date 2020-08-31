import os
from uuid import uuid4

import cv2
import librosa
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as sk
from sklearn.utils import shuffle
from PIL import Image

from tqdm import tqdm

sns.set()
TRAIN_DIR = '/input/'
OUT_DIR = '/output/mel_first/'
DF = '/output/mel_first/samples_df'
IMG_SIZE = 224
SAMPLES = 224
samples_from_file = []


def extract_features_mfcc_mel(file_path, folder_path, name, class_label):
    restored = np.load(file_path, allow_pickle=True)
    wave_data = restored[0][0]
    wave_rate = restored[0][1]
    sample_length = 5 * wave_rate
    i = 0
    for idx in range(0, len(wave_data), sample_length):
        song_sample = wave_data[idx:idx + sample_length]
        if len(song_sample) >= sample_length:
            mel = librosa.feature.melspectrogram(song_sample, n_mels=SAMPLES)
            db = librosa.power_to_db(mel)
            normalised_db = sk.preprocessing.minmax_scale(db)
            filename = str(uuid4()) + ".tif"
            db_array = (np.asarray(normalised_db) * 255).astype(np.uint8)
            db_image = Image.fromarray(np.array([db_array, db_array, db_array]).T)
            db_image.save("{}{}".format(OUT_DIR, filename))
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
                    print('process file ' + audio_file_path + '/' + parts[0] + '.npy')
                    extract_features_mfcc_mel('{}/{}.npy'.format(audio_file_path, parts[0]), audio_file_path, parts[0],
                                              class_label)
            except ZeroDivisionError:
                print("{} is corrupted".format(audio_file_path))
    samples_df = pd.DataFrame(samples_from_file)
    samples_df.to_pickle(DF)
    # df = pd.read_pickle(file_name)


if __name__ == "__main__":
    main()
