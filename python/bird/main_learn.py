import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import EfficientNetB4
from config import BIRD_CODE

warnings.filterwarnings('ignore')
AUTO = tf.data.experimental.AUTOTUNE
sns.set()
TRAIN_DIR = '/input/'
OUT_DIR = '/output/mel/'
EPOCHS = 10
BATCH_SIZE = 32
SAMPLES = 512
SIZE = 224
labels = 264
MODEL_NAME = 'effnet4_mel_noaug'


def build_model():
    inp = tf.keras.layers.Input(shape=(SIZE, SIZE, 3))
    base = EfficientNetB4(input_shape=(SIZE, SIZE, 3), weights='imagenet', include_top=False)
    # base.trainable = False
    x = base(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2, name="top_dropout1")(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2, name="top_dropout2")(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2, name="top_dropout3")(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2, name="top_dropout4")(x)
    # Compile the model
    x = tf.keras.layers.Dense(labels, activation='softmax')(x)
    model = tf.keras.Model(inputs=inp, outputs=x)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', metrics=['AUC'], optimizer=opt)
    return model


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


def create_dataset(df):
    print('create ds from elements ' + str(len(df)))
    ds = tf.data.Dataset.from_tensor_slices((df['song_sample'].values, df['bird'].values))
    ds = ds.map(read_labeled, num_parallel_calls=AUTO)
    # ds = ds.batch(128, drop_remainder=True)
    return ds


def read_labeled_py(rec, rec2):
    print('map record')
    # print(rec)
    # print(rec2)
    # exit(1)
    data = np.load(rec.numpy(), allow_pickle=True)
    return tf.transpose(data), tf.one_hot(BIRD_CODE[rec2.numpy().decode("utf-8")], 264)


def read_labeled(rec, rec2):
    return tf.py_function(read_labeled_py, inp=[rec, rec2], Tout=[tf.float32, tf.float32])


def main():
    # le = LabelEncoder()
    df = pd.read_pickle(OUT_DIR + 'samples_df')
    # df['bird'] = df['bird'].map(lambda x: BIRD_CODE[x])
    # df['bird'] = to_categorical(le.fit_transform(np.array(df.bird.tolist())))
    df = df.sample(frac=1)
    print(df.head())
    train, test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
    train_ds = create_dataset(train)
    val_ds = create_dataset(test)

    model = build_model()
    model.summary()
    history = model.fit(train_ds,
                        epochs=EPOCHS,
                        callbacks=[  # sv,
                            get_lr_callback(BATCH_SIZE)],
                        validation_data=val_ds,  # class_weight = {0:1,1:2},
                        verbose=1,
                        shuffle=True,
                        batch_size=BATCH_SIZE
                        )
    model.save(OUT_DIR + 'model/' + MODEL_NAME)


if __name__ == "__main__":
    main()
