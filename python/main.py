import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications import EfficientNetB5
from tensorflow.keras.applications import EfficientNetB6

from augmentation import count_data_items, get_dataset
from config import IMG_SIZE, SEED, EFF_NET, BATCH_SIZE, INC2019, INC2018, RESIZE_DICT, EPOCHS, MODEL_NAME

AUTO = tf.data.experimental.AUTOTUNE
PATH = '/input/5fold/'
MODEL_PATH = '/output/fold_model/'
VERBOSE = 1
EFNS = [EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3,
        EfficientNetB4, EfficientNetB5, EfficientNetB6]


def build_model(dim=128, ef=0):
    inp = tf.keras.layers.Input(shape=(dim, dim, 3))
    base = EFNS[ef](input_shape=(dim, dim, 3), weights='imagenet', include_top=False)
    # base.trainable = False
    x = base(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # top_dropout_rate = 0.2
    # x = tf.keras.layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inp, outputs=x)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05)
    model.compile(optimizer=opt, loss=loss, metrics=['AUC'])
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


def unfreeze_model(model):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-20:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05)
    model.compile(optimizer=optimizer, loss=loss, metrics=['AUC'])
    return model


def main():
    GCS_PATH = PATH + 'melanoma-%ix%i' % (IMG_SIZE, IMG_SIZE)
    GCS_PATH2 = PATH + 'isic2019-%ix%i' % (IMG_SIZE, IMG_SIZE)
    target_size_ = -1
    for item in RESIZE_DICT:
        if item[0] == EFF_NET:
            target_size_ = item[1]

    print('target size is ' + str(target_size_))
    model = build_model(dim=target_size_, ef=EFF_NET)
    model.summary()
    skf = KFold(n_splits=2, shuffle=True, random_state=SEED)
    for fold, (idxT, idxV) in enumerate(skf.split(np.arange(15))):
        # DISPLAY FOLD INFO
        print('#' * 25)
        print('#### FOLD', fold + 1)
        print('#### Image Size %i with EfficientNet B%i and batch_size %i' %
              (IMG_SIZE, EFF_NET, BATCH_SIZE))
        print('train fold ' + str(idxT))
        print('validation fold ' + str(idxV))

        # CREATE TRAIN AND VALIDATION SUBSETS
        files_train = tf.io.gfile.glob([GCS_PATH + '/train%.2i*.tfrec' % x for x in idxT])
        if INC2019:
            files_train += tf.io.gfile.glob([GCS_PATH2 + '/train%.2i*.tfrec' % x for x in idxT * 2 + 1])
            print('#### Using 2019 external data')
        if INC2018:
            files_train += tf.io.gfile.glob([GCS_PATH2 + '/train%.2i*.tfrec' % x for x in idxT * 2])
            print('#### Using 2018+2017 external data')
        np.random.shuffle(files_train)
        print('#' * 25)
        files_valid = tf.io.gfile.glob([GCS_PATH + '/train%.2i*.tfrec' % x for x in idxV])
        # TRAIN
        print('Training...')
        for epoch in range(0, EPOCHS):
            fold_iteration(files_train, files_valid, model, target_size_, epoch, fold)

    model.save(MODEL_PATH + MODEL_NAME + 'efnet_' + str(EFF_NET) + '_fold-%i-%ix%i.model' % (
        fold, IMG_SIZE, IMG_SIZE))


def fold_iteration(files_train, files_valid, model, target_size_, epoch, fold):
    dataset = get_dataset(files_train, augment=True, shuffle=True, repeat=True, dim=target_size_,
                          batch_size=BATCH_SIZE)
    validation_data = get_dataset(files_valid, augment=False, shuffle=False, repeat=False,
                                  dim=target_size_)
    count_data = count_data_items(files_train) / BATCH_SIZE
    history = model.fit(dataset,
                        epochs=1,
                        callbacks=[  # sv,
                            get_lr_callback(BATCH_SIZE)],
                        steps_per_epoch=count_data,
                        validation_data=validation_data,  # class_weight = {0:1,1:2},
                        verbose=VERBOSE,
                        batch_size=BATCH_SIZE
                        )
    model.save(MODEL_PATH + MODEL_NAME + 'efnet_' + str(EFF_NET) + '_iter' + str(
        epoch) + '_' + '_fold-%i-%ix%i.model' % (
                   fold, IMG_SIZE, IMG_SIZE))


if __name__ == "__main__":
    main()
