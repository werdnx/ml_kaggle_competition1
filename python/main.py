import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.model_selection import KFold
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications import EfficientNetB5
from tensorflow.keras.applications import EfficientNetB6

from augmentation import count_data_items, get_dataset
from config import FOLDS, IMG_SIZES, REPLICAS, SEED, EFF_NETS, BATCH_SIZES, INC2019, \
    INC2018, VERSION, EPOCHS, MODEL_NAME, RESIZE_DICT

AUTO = tf.data.experimental.AUTOTUNE
PATH = '/input/5fold/'
MODEL_PATH = '/output/fold_model/'
VERBOSE = 1
DISPLAY_PLOT = False
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
    lr_max = 0.00000125 * REPLICAS * batch_size
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
    strategy = tf.distribute.get_strategy()

    GCS_PATH = [None] * FOLDS
    GCS_PATH2 = [None] * FOLDS
    for i, k in enumerate(IMG_SIZES):
        GCS_PATH[i] = PATH + 'melanoma-%ix%i' % (k, k)
        GCS_PATH2[i] = PATH + 'isic2019-%ix%i' % (k, k)

    skf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    for fold, (idxT, idxV) in enumerate(skf.split(np.arange(15))):

        # DISPLAY FOLD INFO
        print('#' * 25)
        print('#### FOLD', fold + 1)
        print('#### Image Size %i with EfficientNet B%i and batch_size %i' %
              (IMG_SIZES[fold], EFF_NETS[fold], BATCH_SIZES[fold]))

        # CREATE TRAIN AND VALIDATION SUBSETS
        files_train = tf.io.gfile.glob([GCS_PATH[fold] + '/train%.2i*.tfrec' % x for x in idxT])
        if INC2019[fold]:
            files_train += tf.io.gfile.glob([GCS_PATH2[fold] + '/train%.2i*.tfrec' % x for x in idxT * 2 + 1])
            print('#### Using 2019 external data')
        if INC2018[fold]:
            files_train += tf.io.gfile.glob([GCS_PATH2[fold] + '/train%.2i*.tfrec' % x for x in idxT * 2])
            print('#### Using 2018+2017 external data')
        np.random.shuffle(files_train);
        print('#' * 25)
        files_valid = tf.io.gfile.glob([GCS_PATH[fold] + '/train%.2i*.tfrec' % x for x in idxV])
        files_test = np.sort(np.array(tf.io.gfile.glob(GCS_PATH[fold] + '/test*.tfrec')))

        # TRAIN
        print('Training...')
        target_size_ = -1
        for item in RESIZE_DICT:
            if item[0] == EFF_NETS[fold]:
                target_size_ = item[1]

        # BUILD MODEL
        K.clear_session()
        with strategy.scope():
            model = build_model(dim=target_size_, ef=EFF_NETS[fold])

        # SAVE BEST MODEL EACH FOLD
        sv = tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH + VERSION + 'fold-%i.h5' % fold, monitor='val_loss', verbose=0, save_best_only=True,
            save_weights_only=True, mode='min', save_freq='epoch')

        dataset = get_dataset(files_train, augment=True, shuffle=True, repeat=True, dim=target_size_,
                              batch_size=BATCH_SIZES[fold])
        validation_data = get_dataset(files_valid, augment=False, shuffle=False, repeat=False,
                                      dim=target_size_)
        count_data = count_data_items(files_train) / BATCH_SIZES[fold] // REPLICAS
        history = model.fit(dataset,
                            epochs=EPOCHS[fold],
                            callbacks=[sv, get_lr_callback(BATCH_SIZES[fold])],
                            steps_per_epoch=count_data,
                            validation_data=validation_data,  # class_weight = {0:1,1:2},
                            verbose=VERBOSE,
                            batch_size=BATCH_SIZES[fold]
                            )
        # model = unfreeze_model(model)
        # history = model.fit(dataset,
        #                     epochs=int(float(EPOCHS[fold]) / 3), callbacks=[sv, get_lr_callback(BATCH_SIZES[fold])],
        #                     steps_per_epoch=count_data,
        #                     validation_data=validation_data,  # class_weight = {0:1,1:2},
        #                     verbose=VERBOSE,
        #                     batch_size=BATCH_SIZES[fold]
        #                     )

        model.save(MODEL_PATH + MODEL_NAME + 'fold-%i-%ix%i.model' % (fold, IMG_SIZES[fold], IMG_SIZES[fold]))


if __name__ == "__main__":
    main()
