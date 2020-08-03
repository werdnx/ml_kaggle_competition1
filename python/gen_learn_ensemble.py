import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB5
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers.experimental import preprocessing
import time
import cv2
from utils import normalize_image_tf

from model_params import batch_size, target_size_, batch_size_, epochs_, model_name, label_smooth_fac

NUM_CLASSES = 2


def input_preprocess_test(image, label):
    label = tf.one_hot(label, NUM_CLASSES)
    image = tf.image.resize(image, [target_size_, target_size_], antialias=True)
    image = tf.cast(image, tf.float32) / 255.0

    return image, label


def input_preprocess_train(image, label):
    label = tf.one_hot(label, NUM_CLASSES)
    image = tf.image.resize(image, [target_size_, target_size_], antialias=True)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_saturation(image, 0.7, 1.3)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_brightness(image, 0.1)
    return image, label


def unfreeze_model(model):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-20:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


# ensemble from b3 and b4 without freeze
def run():
    builder = tfds.ImageFolder('/input/jpeg/tsds/')
    print(builder.info)
    time.sleep(10)
    # num examples, labels... are automatically calculated
    ds_train = builder.as_dataset(split='train', shuffle_files=True, as_supervised=True)
    ds_test = builder.as_dataset(split='test', shuffle_files=True, as_supervised=True)
    # size = (target_size_, target_size_)
    # ds_train = ds_train.map(lambda image, label: (tf.image.resize(image, size), label))
    # ds_test = ds_test.map(lambda image, label: (tf.image.resize(image, size), label))
    # img_augmentation = Sequential(
    #     [
    #         preprocessing.RandomRotation(factor=0.15),
    #         preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
    #         preprocessing.RandomFlip(),
    #         # preprocessing.RandomZoom(height_factor=(-0.3, -0.2), width_factor=(-0.3, -0.2)),
    #         preprocessing.RandomContrast(factor=0.1),
    #         preprocessing.Rescaling(scale=1. / 255.)
    #     ],
    #     name="img_augmentation"
    # )
    ds_train = ds_train.map(
        input_preprocess_train, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    ds_train = ds_train.batch(batch_size=batch_size_, drop_remainder=True)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.map(input_preprocess_test)
    ds_test = ds_test.batch(batch_size=batch_size, drop_remainder=True)

    model_input = tf.keras.Input(shape=(target_size_, target_size_, 3),
                                 name='img_input')

    dummy = tf.keras.layers.Lambda(lambda x: x)(model_input)

    outputs = []

    x = EfficientNetB3(include_top=False,
                       weights='imagenet',
                       input_shape=(target_size_, target_size_, 3),
                       pooling='avg')(dummy)
    x = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    outputs.append(x)

    x = EfficientNetB4(include_top=False,
                       weights='imagenet',
                       input_shape=(target_size_, target_size_, 3),
                       pooling='avg')(dummy)
    x = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    outputs.append(x)

    # x = EfficientNetB5(include_top=False,
    #                    weights='imagenet',
    #                    input_shape=(target_size_, target_size_, 3),
    #                    pooling='avg')(dummy)
    # x = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    # outputs.append(x)

    model = tf.keras.Model(model_input, outputs, name='aNetwork')
    model.summary()

    # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer='adam', loss=[
            tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=label_smooth_fac),
            tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=label_smooth_fac),
            # tf.keras.losses.CategoricalCrossentropy(
            #     label_smoothing=label_smooth_fac)
        ], metrics=[tf.keras.metrics.AUC(name='auc')]
    )
    # epochs = int(2 * (float(epochs_) / 3))  # @param {type: "slider", min:8, max:80}
    epochs = epochs_
    hist = model.fit(ds_train, epochs=epochs, validation_data=ds_test, verbose=1)
    model.save('/output/' + model_name)
    # model = unfreeze_model(model)
    # epochs = int((float(epochs_) / 3))  # @param {type: "slider", min:8, max:50}
    # hist = model.fit(ds_train, epochs=epochs, validation_data=ds_test, verbose=2)
    # model.save('/output/' + model_name)
