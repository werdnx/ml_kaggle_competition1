import os
import cv2
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from shutil import copyfile
import cv2
from augmentation import size

# from os import cpu_count
# from multiprocessing.pool import Pool
# from functools import partial
IN_DIR = '/home/werdn/input/jpeg/train512_nohair/'
IN_DIR_TEST = '/home/werdn/input/jpeg/test512_nohair/'
OUT_DIR_TEST = '/home/werdn/input/jpeg/test512_nohair_normalized/'
OUT_DIR = '/home/werdn/input/jpeg/tsds/'


# test=validation

def normalize(path, out_path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(size, size))
    image[:, :, 0] = clahe.apply(image[:, :, 0])
    cv2.imwrite(out_path, image)


def main():
    train_df = pd.read_csv('/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1/input/train.csv')
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    train, test = train_test_split(train_df, test_size=0.3)
    test_images = [(IN_DIR_TEST + i, i) for i in os.listdir(IN_DIR_TEST)]
    for i, image_file in enumerate(test_images):
        print ("write file " + OUT_DIR_TEST + image_file[1])
        normalize(image_file[0], OUT_DIR_TEST + image_file[1])

    for index, row in train.iterrows():
        if str(row['target']) == "1":
            normalize(IN_DIR + row['image_name'] + '.jpg', OUT_DIR + 'train/malignant/' + row['image_name'] + '.jpg')
            # copyfile(IN_DIR + row['image_name'] + '.jpg', OUT_DIR + 'train/malignant/' + row['image_name'] + '.jpg')
            print ("write file " + OUT_DIR + 'train/malignant/' + row['image_name'] + '.jpg')
        else:
            normalize(IN_DIR + row['image_name'] + '.jpg', OUT_DIR + 'train/benign/' + row['image_name'] + '.jpg')
            # copyfile(IN_DIR + row['image_name'] + '.jpg', OUT_DIR + 'train/benign/' + row['image_name'] + '.jpg')
            print ("write file " + OUT_DIR + 'train/benign/' + row['image_name'] + '.jpg')
    for index, row in test.iterrows():
        if str(row['target']) == "1":
            normalize(IN_DIR + row['image_name'] + '.jpg', OUT_DIR + 'test/malignant/' + row['image_name'] + '.jpg')
            # copyfile(IN_DIR + row['image_name'] + '.jpg', OUT_DIR + 'test/malignant/' + row['image_name'] + '.jpg')
            print ("write file " + OUT_DIR + 'test/malignant/' + row['image_name'] + '.jpg')
        else:
            normalize(IN_DIR + row['image_name'] + '.jpg', OUT_DIR + 'test/benign/' + row['image_name'] + '.jpg')
            # copyfile(IN_DIR + row['image_name'] + '.jpg', OUT_DIR + 'test/benign/' + row['image_name'] + '.jpg')
            print ("write file " + OUT_DIR + 'test/benign/' + row['image_name'] + '.jpg')


if __name__ == "__main__":
    main()
