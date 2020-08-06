import pandas as pd, numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications import EfficientNetB5
from tensorflow.keras.applications import EfficientNetB6
from keras.models import load_model

from augmentation import count_data_items, get_dataset
from config import FOLDS, IMG_SIZES, REPLICAS, SEED, DEVICE, EFF_NETS, BATCH_SIZES, INC2019, \
    INC2018, VERSION, EPOCHS, TTA, WGTS, MODEL_NAME

from main import MODEL_PATH

from main import PATH

from main import VERBOSE


def main():
    GCS_PATH = [None] * FOLDS
    for i, k in enumerate(IMG_SIZES):
        GCS_PATH[i] = PATH + 'melanoma-%ix%i' % (k, k)

    skf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    files_test = np.sort(np.array(tf.io.gfile.glob(GCS_PATH[0] + '/test*.tfrec')))
    preds = np.zeros((count_data_items(files_test), 1))

    for fold, (idxT, idxV) in enumerate(skf.split(np.arange(15))):
        print('#' * 25)
        print('#### FOLD', fold + 1)
        print('#### Image Size %i with EfficientNet B%i and batch_size %i' %
              (IMG_SIZES[fold], EFF_NETS[fold], BATCH_SIZES[fold]))

        files_test = np.sort(np.array(tf.io.gfile.glob(GCS_PATH[fold] + '/test*.tfrec')))

        model = load_model(MODEL_PATH + MODEL_NAME + 'fold-%i-%ix%i.model' % (fold, IMG_SIZES[fold], IMG_SIZES[fold]))
        model.summary()

        # PREDICT TEST USING TTA
        print('Predicting Test with TTA...')
        ds_test = get_dataset(files_test, labeled=False, return_image_names=False, augment=True,
                              repeat=True, shuffle=False, dim=IMG_SIZES[fold], batch_size=BATCH_SIZES[fold])
        ct_test = count_data_items(files_test)
        STEPS = TTA * ct_test / BATCH_SIZES[fold] / 4 / REPLICAS
        pred = model.predict(ds_test, steps=STEPS, verbose=VERBOSE)[:TTA * ct_test, ]
        # preds[:, 0] += np.mean(pred.reshape((ct_test, TTA), order='F'), axis=1) * WGTS[fold]
        print(pred)

if __name__ == "__main__":
    main()
