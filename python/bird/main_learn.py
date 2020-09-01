import random
import warnings

import librosa
import keras
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as sk
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0
from tqdm import tqdm

from config import BIRD_CODE

warnings.filterwarnings('ignore')
AUTO = tf.data.experimental.AUTOTUNE
sns.set()
TRAIN_DIR = '/input/'
OUT_DIR = '/output/'
EPOCHS = 100
BATCH_SIZE = 32
EARLY_STOP_PATIENCE = 10
# SAMPLES = 512
# SIZE = 224
labels = 264
MODEL_NAME = 'conv1d_fft_model_v1'
DF = '/input/sample_slides/samples_df'
SAMPLES_RESTRICTION = 2000
SAMPLING_RATE = 22050


# model.add(effnet_layers)
#
# model.add(GlobalAveragePooling2D())
# model.add(Dense(256, use_bias=False))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dropout(dropout_dense_layer))

def add_noise(data, noise_factor, p):
    if random.random() <= p:
        noise = np.random.randn(len(data))
        augmented_data = data + noise_factor * noise
        # Cast back to same data type
        augmented_data = augmented_data.astype(type(data[0]))
        return augmented_data
    else:
        return data


def shift(data, sampling_rate, shift_max, shift_direction, p):
    if random.random() <= p:
        shift = np.random.randint(sampling_rate * shift_max)
        if shift_direction == 'right':
            shift = -shift
        elif shift_direction == 'both':
            direction = np.random.randint(0, 2)
            if direction == 1:
                shift = -shift
        augmented_data = np.roll(data, shift)
        # Set to silence for heading/ tailing
        if shift > 0:
            augmented_data[:shift] = 0
        else:
            augmented_data[shift:] = 0
        return augmented_data
    else:
        return data


def change_pitch(data, sampling_rate, pitch_factor, p):
    if random.random() <= p:
        return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)
    else:
        return data


def change_speed(data, speed_factor, p):
    if random.random() <= p:
        return librosa.effects.time_stretch(data, speed_factor)
    else:
        return data


def residual_block(x, filters, conv_num=3, activation="relu"):
    # Shortcut
    s = keras.layers.Conv1D(filters, 1, padding="same")(x)
    for i in range(conv_num - 1):
        x = keras.layers.Conv1D(filters, 3, padding="same")(x)
        x = keras.layers.Activation(activation)(x)
    x = keras.layers.Conv1D(filters, 3, padding="same")(x)
    x = keras.layers.Add()([x, s])
    x = keras.layers.Activation(activation)(x)
    return keras.layers.MaxPool1D(pool_size=2, strides=2)(x)


def build_model(input_shape, num_classes):
    inputs = keras.layers.Input(shape=input_shape, name="input")

    x = residual_block(inputs, 16, 2)
    x = residual_block(x, 32, 2)
    x = residual_block(x, 64, 3)
    x = residual_block(x, 128, 3)
    x = residual_block(x, 128, 3)

    x = keras.layers.AveragePooling1D(pool_size=3, strides=3)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dense(128, activation="relu")(x)

    outputs = keras.layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.models.Model(inputs=inputs, outputs=outputs)

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    # metrics=['AUC']
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)
    return model


#
# def build_model():
#     inp = tf.keras.layers.Input(shape=(SIZE, SIZE, 3))
#     base = EfficientNetB0(input_shape=(SIZE, SIZE, 3), weights='imagenet', include_top=False)
#     # base.trainable = False
#     x = base(inp)
#     x = tf.keras.layers.GlobalAveragePooling2D()(x)
#     # x = tf.keras.layers.Dense(256, use_bias=False)(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.Dropout(0.25, name="top_dropout1")(x)
#     # x = tf.keras.layers.Dense(512, activation='relu')(x)
#     # x = tf.keras.layers.Dropout(0.2, name="top_dropout2")(x)
#     # x = tf.keras.layers.Dense(256, activation='relu')(x)
#     # x = tf.keras.layers.Dropout(0.2, name="top_dropout3")(x)
#     # x = tf.keras.layers.Dense(128, activation='relu')(x)
#     # x = tf.keras.layers.Dropout(0.2, name="top_dropout4")(x)
#     # x = tf.keras.layers.Dense(64, activation='relu')(x)
#     # Compile the model
#     x = tf.keras.layers.Dense(labels, activation='softmax')(x)
#     model = tf.keras.Model(inputs=inp, outputs=x)
#     opt = tf.keras.optimizers.Adam(learning_rate=0.001)
#     model.compile(loss='categorical_crossentropy', metrics=['AUC'], optimizer=opt)
#     return model


def get_lr_callback(batch_size=8):
    lr_start = 0.000005
    lr_max = 0.00000125 * batch_size
    lr_min = 0.000001
    lr_ramp_ep = 5
    lr_sus_ep = 0
    lr_decay = 0.8

    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) * lr_decay ** (epoch - lr_ramp_ep - lr_sus_ep) + lr_min
        return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
    return lr_callback


def create_dataset(df, augument):
    print('create ds from elements ' + str(len(df)))
    ds = tf.data.Dataset.from_tensor_slices((df['song_sample'].values, df['bird'].values))
    ds = ds.map(lambda rec1, rec2: read_labeled(rec1, rec2, augument), num_parallel_calls=AUTO)
    if augument:
        ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE, drop_remainder=True)
    return ds


def audio_to_fft(audio):
    fft = tf.signal.fft(
        tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
    )
    return tf.math.abs(fft[: (audio.shape[0] // 2), :])


def read_labeled_py(rec, rec2, augument):
    # print('read file from path ' + str(rec.numpy()))
    restored = np.load(rec.numpy().decode("utf-8"), allow_pickle=True)
    wave_data = restored[0][0]
    wave_rate = restored[0][1]
    if augument:
        wave_data = add_noise(wave_data, random.uniform(-0.001, 0.001), 0.9)
        wave_data = shift(wave_data, wave_rate, random.uniform(0.25, 0.5), 'both', 0.8)
        wave_data = change_pitch(wave_data, wave_rate, random.uniform(0.5, 5), 0.7)
        wave_data = change_speed(wave_data, random.uniform(0.85, 1.2), 0.6)

    wave_data = tf.reshape(wave_data, [wave_data.shape[0], 1])
    data = audio_to_fft(wave_data)
    return data, tf.one_hot(BIRD_CODE[rec2.numpy().decode("utf-8")], labels)


def read_labeled(rec, rec2, augument):
    return tf.py_function(read_labeled_py, inp=[rec, rec2, augument], Tout=[tf.float32, tf.float32])


def adjust_data(df):
    df = df.sample(frac=1)
    samples = []
    b_uniq = df.bird.unique()
    with tqdm(total=len(b_uniq)) as pbar:
        for bird in b_uniq:
            pbar.update(1)
            bird_df = df[df['bird'] == bird]
            l = len(bird_df)
            if l > SAMPLES_RESTRICTION:
                for i in range(0, SAMPLES_RESTRICTION):
                    samples.append(
                        {"song_sample": "{}".format(bird_df.iloc[i]['song_sample']), "bird": bird_df.iloc[i]['bird']})
            elif l < SAMPLES_RESTRICTION:
                for i in range(0, l):
                    samples.append(
                        {"song_sample": "{}".format(bird_df.iloc[i]['song_sample']), "bird": bird_df.iloc[i]['bird']})
                for i in range(l, SAMPLES_RESTRICTION):
                    ii = random.randint(0, l - 1)
                    samples.append(
                        {"song_sample": "{}".format(bird_df.iloc[ii]['song_sample']),
                         "bird": bird_df.iloc[ii]['bird']})

    res = pd.DataFrame(samples)
    res = res.sample(frac=1)
    return res


def main():
    # le = LabelEncoder()
    df = pd.read_pickle(DF)
    df = adjust_data(df)
    df.head()
    # df['bird'] = df['bird'].map(lambda x: BIRD_CODE[x])
    # df['bird'] = to_categorical(le.fit_transform(np.array(df.bird.tolist())))
    df = df.sample(frac=1)
    print(df.head())
    train, test = train_test_split(df, test_size=0.3, random_state=42, shuffle=True)
    count_data = len(train) / BATCH_SIZE
    train_ds = create_dataset(train, True)
    val_ds = create_dataset(test, False)

    model = build_model((SAMPLING_RATE // 2, 1), labels)
    model.summary()
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=EARLY_STOP_PATIENCE)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=OUT_DIR + 'model/best_bird_fft_model.h5',
                                                    monitor='val_loss',
                                                    save_best_only=True)
    history = model.fit(train_ds,
                        epochs=EPOCHS,
                        callbacks=[  # sv,
                            get_lr_callback(BATCH_SIZE), early_stop, checkpoint],
                        validation_data=val_ds,  # class_weight = {0:1,1:2},
                        steps_per_epoch=count_data,
                        verbose=1,
                        batch_size=BATCH_SIZE
                        )
    model.save(OUT_DIR + 'model/' + MODEL_NAME)


if __name__ == "__main__":
    main()
