import os
import warnings

import librosa
import noisereduce as nr
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras.utils import to_categorical
from numpy import save
from scipy.ndimage import maximum_filter1d
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from tqdm import tqdm

warnings.filterwarnings('ignore')

print(librosa.__version__)

sns.set()
TRAIN_DIR = '/input/'
EPOCHS = 10
BATCH_SIZE = 32
SAMPLES = 512


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
                if os.path.exists(audio_file_path + '/' + parts[0] + '.npy'):
                    print('skip file ' + audio_file_path + '/' + parts[0] + '.npy')
                    # data = np.load(audio_file_path + '/' + parts[0] + '.npy')
                    # features.append([data, class_label])
                else:
                    # print('file does no EXIST ' + audio_file_path + '/' + parts[0] + '.npy')
                    print('procces file ' + '{}/{}'.format(audio_file_path, row.filename))
                    data = extract_features('{}/{}'.format(audio_file_path, row.filename))
                    save_to_file(data, audio_file_path, parts[0])
                    features.append([data, class_label])
            except ZeroDivisionError:
                print("{} is corrupted".format(audio_file_path))

    featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])
    featuresdf.head()
    X = np.array(featuresdf.feature.tolist())
    y = np.array(featuresdf.class_label.tolist())
    le = LabelEncoder()
    yy = to_categorical(le.fit_transform(y))
    x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state=42)
    num_labels = yy.shape[1]
    model = build_model_graph(num_labels)

    # model.summary()
    # Calculate pre-training accuracy
    # score = model.evaluate(x_test, y_test, verbose=0)
    # accuracy = 100 * score[1]
    # print("Pre-training accuracy: %.4f%%" % accuracy)

    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_test, y_test),
              verbose=1)
    # Evaluating the model on the training and testing set
    score = model.evaluate(x_train, y_train, verbose=1)
    print("Training Accuracy: {0:.2%}".format(score[1]))
    score = model.evaluate(x_test, y_test, verbose=1)
    print("Testing Accuracy: {0:.2%}".format(score[1]))


if __name__ == "__main__":
    main()
