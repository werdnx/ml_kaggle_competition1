import random
import warnings

import albumentations as A
import cv2
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0
from tqdm import tqdm

from config import BIRD_CODE

warnings.filterwarnings('ignore')
AUTO = tf.data.experimental.AUTOTUNE
sns.set()
TRAIN_DIR = '/input/'
OUT_DIR = '/output/'
EPOCHS = 20
BATCH_SIZE = 16
SAMPLES = 512
SIZE = 224
labels = 264
MODEL_NAME = 'effnet4_mel_noaug'
SAMPLES_RESTRICTION = 1000


# model.add(effnet_layers)
#
# model.add(GlobalAveragePooling2D())
# model.add(Dense(256, use_bias=False))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dropout(dropout_dense_layer))


def build_model():
    inp = tf.keras.layers.Input(shape=(SIZE, SIZE, 3))
    base = EfficientNetB0(input_shape=(SIZE, SIZE, 3), weights='imagenet', include_top=False)
    # base.trainable = False
    x = base(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # x = tf.keras.layers.Dense(256, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2, name="top_dropout1")(x)
    # x = tf.keras.layers.Dense(512, activation='relu')(x)
    # x = tf.keras.layers.Dropout(0.2, name="top_dropout2")(x)
    # x = tf.keras.layers.Dense(256, activation='relu')(x)
    # x = tf.keras.layers.Dropout(0.2, name="top_dropout3")(x)
    # x = tf.keras.layers.Dense(128, activation='relu')(x)
    # x = tf.keras.layers.Dropout(0.2, name="top_dropout4")(x)
    # x = tf.keras.layers.Dense(64, activation='relu')(x)
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


def create_dataset(df, augument):
    print('create ds from elements ' + str(len(df)))
    ds = tf.data.Dataset.from_tensor_slices((df['song_sample'].values, df['bird'].values))
    ds = ds.map(lambda rec1, rec2: read_labeled(rec1, rec2, augument), num_parallel_calls=AUTO)
    if augument:
        ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE, drop_remainder=True)
    return ds


def read_labeled_py(rec, rec2, augument):
    # print('read file from path ' + str(rec.numpy()))
    img = cv2.imread(rec.numpy().decode("utf-8"))
    if augument:
        albus = A.Cutout(max_h_size=int(img.shape[0] * 0.375), max_w_size=int(img.shape[1] * 0.375), num_holes=1, p=0.7)
        img = albus(image=img)['image']

    img = tf.image.resize(img, [SIZE, SIZE], antialias=True)
    img = tf.cast(img, tf.float32) / 255.0
    return img, tf.one_hot(BIRD_CODE[rec2.numpy().decode("utf-8")], 264)


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
    df = pd.read_pickle(OUT_DIR + 'mel_first/samples_df')
    df = adjust_data(df)
    df.head()
    # df['bird'] = df['bird'].map(lambda x: BIRD_CODE[x])
    # df['bird'] = to_categorical(le.fit_transform(np.array(df.bird.tolist())))
    df = df.sample(frac=1)
    print(df.head())
    count_data = len(df) / BATCH_SIZE
    train, test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
    train_ds = create_dataset(train, True)
    val_ds = create_dataset(test, False)

    model = build_model()
    model.summary()
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=OUT_DIR + 'model/best_bird_model.h5', monitor='val_loss', save_best_only=True)
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
