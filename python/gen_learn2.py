import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers.experimental import preprocessing
import time

from model_params import batch_size, target_size_, batch_size_, epochs_, model_name

NUM_CLASSES = 2


def input_preprocess(image, label):
    label = tf.one_hot(label, NUM_CLASSES)
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


def run():
    builder = tfds.ImageFolder('/input/jpeg/tsds/')
    print(builder.info)
    time.sleep(10)
    # num examples, labels... are automatically calculated
    ds_train = builder.as_dataset(split='train', shuffle_files=True, as_supervised=True)
    ds_test = builder.as_dataset(split='test', shuffle_files=True, as_supervised=True)
    size = (target_size_, target_size_)
    ds_train = ds_train.map(lambda image, label: (tf.image.resize(image, size), label))
    ds_test = ds_test.map(lambda image, label: (tf.image.resize(image, size), label))
    img_augmentation = Sequential(
        [
            preprocessing.RandomRotation(factor=0.15),
            preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
            preprocessing.RandomFlip(),
            preprocessing.RandomZoom(height_factor=(-0.3, -0.2), width_factor=(-0.3, -0.2)),
            #preprocessing.RandomContrast(factor=0.1)
            #, preprocessing.Rescaling(scale=1. / 255.)
        ],
        name="img_augmentation"
    )
    ds_train = ds_train.map(
        input_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    ds_train = ds_train.batch(batch_size=batch_size_, drop_remainder=True)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.map(input_preprocess)
    ds_test = ds_test.batch(batch_size=batch_size, drop_remainder=True)

    inputs = layers.Input(shape=(target_size_, target_size_, 3))
    x = img_augmentation(inputs)
    model = EfficientNetB3(include_top=False, input_tensor=x, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    epochs = int(2 * (float(epochs_) / 3))  # @param {type: "slider", min:8, max:80}
    hist = model.fit(ds_train, epochs=epochs, validation_data=ds_test, verbose=1)
    model.save('/output/freeze_' + model_name)
    model = unfreeze_model(model)
    epochs = int((float(epochs_) / 3))  # @param {type: "slider", min:8, max:50}
    hist = model.fit(ds_train, epochs=epochs, validation_data=ds_test, verbose=1)
    model.save('/output/' + model_name)
