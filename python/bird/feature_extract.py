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

warnings.filterwarnings('ignore')

print(librosa.__version__)

sns.set()
TRAIN_DIR = '/input/'
EPOCHS = 10
BATCH_SIZE = 32
SAMPLES = 256


def audio_to_spec(audio, sr):
    spec = librosa.power_to_db(
        librosa.feature.melspectrogram(audio, sr=sr, fmin=20, fmax=16000, n_mels=128)
    )
    return spec.astype(np.float32)


def envelope(y, rate, threshold):
    mask = []
    y_mean = maximum_filter1d(np.abs(y), mode="constant", size=rate // 20)
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask, y_mean


def denoise(audio, sr):
    thr = 0.25
    mask, env = envelope(audio, sr, thr)
    return nr.reduce_noise(audio_clip=audio, noise_clip=audio[np.logical_not(mask)], verbose=True)


def extract_features(file_name):
    audio, sample_rate = librosa.load(os.fspath(file_name))
    # audio = denoise(audio, sample_rate)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=SAMPLES)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    S_mel = librosa.feature.melspectrogram(audio, sr=sample_rate)
    S_DB_mel = librosa.amplitude_to_db(S_mel, ref=np.max)
    mel_processed = np.mean(S_DB_mel.T, axis=0)
    return np.append(mfccs_processed, mel_processed)


## Пройтись по каждому файлу
## нарезать файл по 5 сек
def extract_features_mfcc_mel(file_path, folder_path, name):
    # wave_data, wave_rate = librosa.load(file_name)
    if os.path.exists(folder_path + '/' + name + '_mfccs_m.npy') or not os.path.exists(file_path):
        return
    else:
        restored = np.load(file_path, allow_pickle=True)
        wave_data = restored[0][0]
        wave_rate = restored[0][1]
        # wave_data, _ = librosa.effects.trim(wave_data)
        # only take 5s samples and add them to the dataframe
        sample_length = 5 * wave_rate
        i = 0
        correlation_arr = []
        for idx in range(0, len(wave_data), sample_length):
            song_sample = wave_data[idx:idx + sample_length]
            if len(song_sample) >= sample_length:
                correlation_arr.append([])
                # mel = librosa.feature.melspectrogram(song_sample, n_mels=SAMPLES)
                # db = librosa.power_to_db(mel)
                mfccs = librosa.feature.mfcc(y=song_sample, sr=wave_rate, n_mfcc=SAMPLES)
                mfccs_processed = np.mean(mfccs.T, axis=0)
                correlation_arr[i].append(mfccs_processed)
                i = i + 1
                # db_name = name + '_db_' + str(idx)
                # save_to_file(db, folder_path, db_name)
                # mfccs_name = name + '_mfccs_' + str(idx)
                # path = save_to_file(mfccs, folder_path, mfccs_name)
                # correlation_arr = np.append(correlation_arr, (class_label, path))
        save_to_file(correlation_arr, folder_path, name + '_mfccs_m')


def extract_waves(file_name, folder_path, name):
    wave_data, wave_rate = librosa.load(file_name)
    wave_data, _ = librosa.effects.trim(wave_data)
    data = [(wave_data, wave_rate)]
    save_to_file(data, folder_path, name)


def build_model_graph(num_labels, input_shape=(40,)):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(256))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(256))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(num_labels))
    model.add(tf.keras.layers.Activation('softmax'))
    # Compile the model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    return model


def save_to_file(data, audio_file_path, file_name):
    path = audio_file_path + '/' + file_name + '.npy'
    save(path, data)
    return path


# print('save file ' + path)


# load array
# data = load('data.npy')


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
                parts = row.filename.split(".")
                # if os.path.exists(audio_file_path + '/' + parts[0] + '.npy'):
                #     print('skip file ' + audio_file_path + '/' + parts[0] + '.npy')
                # data = np.load(audio_file_path + '/' + parts[0] + '.npy')
                # features.append([data, class_label])
                # else:
                # print('file does no EXIST ' + audio_file_path + '/' + parts[0] + '.npy')
                # print('procces file ' + '{}/{}'.format(audio_file_path, row.filename))
                # extract_waves('{}/{}'.format(audio_file_path, row.filename), audio_file_path, parts[0])
                extract_features_mfcc_mel('{}/{}.npy'.format(audio_file_path, parts[0]), audio_file_path, parts[0])
                # save_to_file(data, audio_file_path, parts[0])
                # features.append([data, class_label])
            except ZeroDivisionError:
                print("{} is corrupted".format(audio_file_path))


if __name__ == "__main__":
    main()
