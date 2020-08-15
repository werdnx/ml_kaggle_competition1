import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications import EfficientNetB5
from tensorflow.keras.applications import EfficientNetB6

from augmentation import count_data_items, get_dataset
from python.old.kfold.config_kfold import FOLDS, IMG_SIZES, REPLICAS, SEED, EFF_NETS, BATCH_SIZES, INC2019, \
    INC2018, VERSION, EPOCHS, TTA, WGTS

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
    x = base(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
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


def main():
    strategy = tf.distribute.get_strategy()

    GCS_PATH = [None] * FOLDS
    GCS_PATH2 = [None] * FOLDS
    for i, k in enumerate(IMG_SIZES):
        GCS_PATH[i] = PATH + 'melanoma-%ix%i' % (k, k)
        GCS_PATH2[i] = PATH + 'isic2019-%ix%i' % (k, k)
    # files_train = np.sort(np.array(tf.io.gfile.glob(GCS_PATH[0] + '/train*.tfrec')))
    files_test = np.sort(np.array(tf.io.gfile.glob(GCS_PATH[0] + '/test*.tfrec')))[:10]

    skf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    oof_pred = []
    oof_tar = []
    oof_val = []
    oof_names = []
    oof_folds = []
    preds = np.zeros((count_data_items(files_test), 1))

    for fold, (idxT, idxV) in enumerate(skf.split(np.arange(15))):

        # DISPLAY FOLD INFO
        # if DEVICE == 'TPU':
        #     if tpu: tf.tpu.experimental.initialize_tpu_system(tpu)
        print('#' * 25);
        print('#### FOLD', fold + 1)
        print('#### Image Size %i with EfficientNet B%i and batch_size %i' %
              (IMG_SIZES[fold], EFF_NETS[fold], BATCH_SIZES[fold]))

        # CREATE TRAIN AND VALIDATION SUBSETS
        files_train = tf.io.gfile.glob([GCS_PATH[fold] + '/train%.2i*.tfrec' % x for x in idxT])[:10]
        if INC2019[fold]:
            files_train += tf.io.gfile.glob([GCS_PATH2[fold] + '/train%.2i*.tfrec' % x for x in idxT * 2 + 1])[:10]
            print('#### Using 2019 external data')
        if INC2018[fold]:
            files_train += tf.io.gfile.glob([GCS_PATH2[fold] + '/train%.2i*.tfrec' % x for x in idxT * 2])[:10]
            print('#### Using 2018+2017 external data')
        np.random.shuffle(files_train);
        print('#' * 25)
        files_valid = tf.io.gfile.glob([GCS_PATH[fold] + '/train%.2i*.tfrec' % x for x in idxV])[:10]
        files_test = np.sort(np.array(tf.io.gfile.glob(GCS_PATH[fold] + '/test*.tfrec')))[:10]

        # BUILD MODEL
        K.clear_session()
        with strategy.scope():
            model = build_model(dim=IMG_SIZES[fold], ef=EFF_NETS[fold])

        # SAVE BEST MODEL EACH FOLD
        sv = tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH + VERSION + 'fold-%i.h5' % fold, monitor='val_loss', verbose=0, save_best_only=True,
            save_weights_only=True, mode='min', save_freq='epoch')

        # TRAIN
        print('Training...')
        history = model.fit(
            get_dataset(files_train, augment=True, shuffle=True, repeat=True,
                        dim=IMG_SIZES[fold], batch_size=BATCH_SIZES[fold]),
            epochs=EPOCHS[fold], callbacks=[sv, get_lr_callback(BATCH_SIZES[fold])],
            steps_per_epoch=count_data_items(files_train) / BATCH_SIZES[fold] // REPLICAS,
            validation_data=get_dataset(files_valid, augment=False, shuffle=False,
                                        repeat=False, dim=IMG_SIZES[fold]),  # class_weight = {0:1,1:2},
            verbose=VERBOSE
        )

        print('Loading best model...')
        model.load_weights(MODEL_PATH + VERSION + 'fold-%i.h5' % fold)

        # PREDICT OOF USING TTA
        print('Predicting OOF with TTA...')
        ds_valid = get_dataset(files_valid, labeled=False, return_image_names=False, augment=True,
                               repeat=True, shuffle=False, dim=IMG_SIZES[fold], batch_size=BATCH_SIZES[fold] * 4)
        ct_valid = count_data_items(files_valid)
        STEPS = TTA * ct_valid / BATCH_SIZES[fold] / 4 / REPLICAS
        pred = model.predict(ds_valid, steps=STEPS, verbose=VERBOSE)[:TTA * ct_valid, ]
        oof_pred.append(np.mean(pred.reshape((ct_valid, TTA), order='F'), axis=1))
        # oof_pred.append(model.predict(get_dataset(files_valid,dim=IMG_SIZES[fold]),verbose=1))

        # GET OOF TARGETS AND NAMES
        ds_valid = get_dataset(files_valid, augment=False, repeat=False, dim=IMG_SIZES[fold],
                               labeled=True, return_image_names=True)
        oof_tar.append(np.array([target.numpy() for img, target in iter(ds_valid.unbatch())]))
        oof_folds.append(np.ones_like(oof_tar[-1], dtype='int8') * fold)
        ds = get_dataset(files_valid, augment=False, repeat=False, dim=IMG_SIZES[fold],
                         labeled=False, return_image_names=True)
        oof_names.append(np.array([img_name.numpy().decode("utf-8") for img, img_name in iter(ds.unbatch())]))

        # PREDICT TEST USING TTA
        print('Predicting Test with TTA...')
        ds_test = get_dataset(files_test, labeled=False, return_image_names=False, augment=True,
                              repeat=True, shuffle=False, dim=IMG_SIZES[fold], batch_size=BATCH_SIZES[fold] * 4)
        ct_test = count_data_items(files_test);
        STEPS = TTA * ct_test / BATCH_SIZES[fold] / 4 / REPLICAS
        pred = model.predict(ds_test, steps=STEPS, verbose=VERBOSE)[:TTA * ct_test, ]
        preds[:, 0] += np.mean(pred.reshape((ct_test, TTA), order='F'), axis=1) * WGTS[fold]

        # REPORT RESULTS
        auc = roc_auc_score(oof_tar[-1], oof_pred[-1])
        oof_val.append(np.max(history.history['val_auc']))
        print('#### FOLD %i OOF AUC without TTA = %.3f, with TTA = %.3f' % (fold + 1, oof_val[-1], auc))

        # PLOT TRAINING
        if DISPLAY_PLOT:
            plt.figure(figsize=(15, 5))
            plt.plot(np.arange(EPOCHS[fold]), history.history['auc'], '-o', label='Train AUC', color='#ff7f0e')
            plt.plot(np.arange(EPOCHS[fold]), history.history['val_auc'], '-o', label='Val AUC', color='#1f77b4')
            x = np.argmax(history.history['val_auc']);
            y = np.max(history.history['val_auc'])
            xdist = plt.xlim()[1] - plt.xlim()[0];
            ydist = plt.ylim()[1] - plt.ylim()[0]
            plt.scatter(x, y, s=200, color='#1f77b4');
            plt.text(x - 0.03 * xdist, y - 0.13 * ydist, 'max auc\n%.2f' % y, size=14)
            plt.ylabel('AUC', size=14);
            plt.xlabel('Epoch', size=14)
            plt.legend(loc=2)
            plt2 = plt.gca().twinx()
            plt2.plot(np.arange(EPOCHS[fold]), history.history['loss'], '-o', label='Train Loss', color='#2ca02c')
            plt2.plot(np.arange(EPOCHS[fold]), history.history['val_loss'], '-o', label='Val Loss', color='#d62728')
            x = np.argmin(history.history['val_loss']);
            y = np.min(history.history['val_loss'])
            ydist = plt.ylim()[1] - plt.ylim()[0]
            plt.scatter(x, y, s=200, color='#d62728');
            plt.text(x - 0.03 * xdist, y + 0.05 * ydist, 'min loss', size=14)
            plt.ylabel('Loss', size=14)
            plt.title('FOLD %i - Image Size %i, EfficientNet B%i, inc2019=%i, inc2018=%i' %
                      (fold + 1, IMG_SIZES[fold], EFF_NETS[fold], INC2019[fold], INC2018[fold]), size=18)
            plt.legend(loc=3)
            plt.show()

    oof = np.concatenate(oof_pred)
    true = np.concatenate(oof_tar)
    names = np.concatenate(oof_names)
    folds = np.concatenate(oof_folds)
    auc = roc_auc_score(true, oof)
    print('Overall OOF AUC with TTA = %.3f' % auc)

    # SAVE OOF TO DISK
    df_oof = pd.DataFrame(dict(
        image_name=names, target=true, pred=oof, fold=folds))
    df_oof.to_csv(MODEL_PATH + VERSION + 'oof.csv', index=False)
    df_oof.head()


if __name__ == "__main__":
    main()
