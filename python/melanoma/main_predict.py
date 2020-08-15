import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model

from augmentation import count_data_items, get_dataset
from config import RESIZE_DICT
from main import MODEL_PATH
from main import PATH

IMG_SIZES = [128, 192, 256, 384, 512, 512]
EFF_NETS = [0, 1, 2, 4, 5, 6]
BATCH_SIZES = [1, 1, 1, 1, 1, 1]
MODELS = [['4fold_efnet_0_fold-0-128x128.model', True],
          ['4fold_efnet_1_fold-0-192x192.model', True],
          ['4fold_efnet_2_fold-0-256x256.model', True],
          ['4fold_efnet_4_fold-1-384x384.model', True],
          ['4fold_efnet_5_iter1__fold-1-512x512.model', True],
          ['b6-512-model_4+4+4_epoch', False]]


def main():
    GCS_PATH = [None] * len(IMG_SIZES)
    for i, k in enumerate(IMG_SIZES):
        GCS_PATH[i] = PATH + 'melanoma-%ix%i' % (k, k)
        print('GCS_PATH[i] = ' + GCS_PATH[i])

    files_test = np.sort(np.array(tf.io.gfile.glob(GCS_PATH[0] + '/test*.tfrec')))
    preds = np.zeros((count_data_items(files_test), 1))

    # for i in range(0, 5):
    for i in range(0, 5):
        print('#' * 25)
        print('#### ITERATION', i + 1)
        print('#### Image Size %i with EfficientNet B%i and batch_size %i' %
              (IMG_SIZES[i], EFF_NETS[i], BATCH_SIZES[i]))

        files_test = np.sort(np.array(tf.io.gfile.glob(GCS_PATH[i] + '/test*.tfrec')))

        target_size_ = IMG_SIZES[i]
        if MODELS[i][1]:
            print('##### USE RESIZE FOR model ', MODELS[i][0])
            for item in RESIZE_DICT:
                if item[0] == EFF_NETS[i]:
                    print('##### NEW SIZE IS  ', item[1])
                    target_size_ = item[1]

        model = load_model(MODEL_PATH + str(MODELS[i][0]))
        model.summary()

        # PREDICT TEST USING TTA
        print('Predicting Test with TTA...')
        ds_test = get_dataset(files_test, labeled=False, return_image_names=True, augment=False,
                              repeat=False, shuffle=False, dim=target_size_, batch_size=BATCH_SIZES[i])

        test_images_ds = ds_test.map(lambda image, image_name: image)
        prob = model.predict(test_images_ds)

        test_ids_ds = ds_test.map(lambda image, image_name: image_name).unbatch()
        test_ids = next(iter(test_ids_ds.batch(count_data_items(files_test)))).numpy().astype(
            'U')  # all in one batch
        pred_df = pd.DataFrame({'image_name': test_ids, 'target': prob[:, 0]})
        sub = pd.read_csv('/input/sample_submission.csv')
        sub.drop('target', inplace=True, axis=1)
        sub = sub.merge(pred_df, on='image_name')
        sub.to_csv('/output/fold_model/submission2_' + str(MODELS[i][0]) + '.csv', index=False)


if __name__ == "__main__":
    main()
