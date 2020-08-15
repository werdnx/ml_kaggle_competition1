import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from sklearn.model_selection import KFold

from augmentation import count_data_items, get_dataset
from python.old.kfold.config_kfold import FOLDS, SEED, BATCH_SIZES, MODEL_NAME, RESIZE_DICT
from main_learn import MODEL_PATH
from main_learn import PATH

IMG_SIZES = [128, 192, 256]
EFF_NETS = [0, 1, 2]


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

        target_size_ = -1
        for item in RESIZE_DICT:
            if item[0] == EFF_NETS[fold]:
                target_size_ = item[1]

        model = load_model(MODEL_PATH + MODEL_NAME + 'fold-%i-%ix%i.model' % (fold, IMG_SIZES[fold], IMG_SIZES[fold]))
        model.summary()

        # PREDICT TEST USING TTA
        print('Predicting Test with TTA...')
        ds_test = get_dataset(files_test, labeled=False, return_image_names=True, augment=False,
                              repeat=False, shuffle=False, dim=target_size_, batch_size=BATCH_SIZES[fold])
        # ct_test = count_data_items(files_test)
        # STEPS = TTA * ct_test / BATCH_SIZES[fold] / 4 / REPLICAS
        # pred = model.predict(ds_test, steps=STEPS, verbose=VERBOSE)[:TTA * ct_test, ]
        # print(pred)
        # print('predds')
        # print(preds)
        test_images_ds = ds_test.map(lambda image, image_name: image)
        prob = model.predict(test_images_ds)

        test_ids_ds = ds_test.map(lambda image, image_name: image_name).unbatch()
        test_ids = next(iter(test_ids_ds.batch(count_data_items(files_test)))).numpy().astype(
            'U')  # all in one batch
        pred_df = pd.DataFrame({'image_name': test_ids, 'target': prob[:, 0]})
        sub = pd.read_csv('/input/sample_submission.csv')
        sub.drop('target', inplace=True, axis=1)
        sub = sub.merge(pred_df, on='image_name')
        sub.to_csv('/output/fold_model/submission_' + str(IMG_SIZES[fold]) + '.csv', index=False)


if __name__ == "__main__":
    main()
