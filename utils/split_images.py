import os
import cv2
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from shutil import copyfile

# from os import cpu_count
# from multiprocessing.pool import Pool
# from functools import partial
IN_DIR = '/home/werdn/input/jpeg/train512_nohair/'
OUT_DIR = '/home/werdn/input/jpeg/tsds/'


def main():
    train_df = pd.read_csv('/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1/input/train.csv')
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    train, test = train_test_split(train_df, test_size=0.2)
    for index, row in train.iterrows():
        if str(row['target']) == "1":
            copyfile(IN_DIR + row['image_name'] + '.jpg', OUT_DIR + 'train/malignant/' + row['image_name'] + '.jpg')
            print ("write file " + OUT_DIR + 'train/malignant/' + row['image_name'] + '.jpg')
        else:
            copyfile(IN_DIR + row['image_name'] + '.jpg', OUT_DIR + 'train/benign/' + row['image_name'] + '.jpg')
            print ("write file " + OUT_DIR + 'train/benign/' + row['image_name'] + '.jpg')
    for index, row in test.iterrows():
        if str(row['target']) == "1":
            copyfile(IN_DIR + row['image_name'] + '.jpg', OUT_DIR + 'test/malignant/' + row['image_name'] + '.jpg')
            print ("write file " + OUT_DIR + 'test/malignant/' + row['image_name'] + '.jpg')
        else:
            copyfile(IN_DIR + row['image_name'] + '.jpg', OUT_DIR + 'test/benign/' + row['image_name'] + '.jpg')
            print ("write file " + OUT_DIR + 'test/benign/' + row['image_name'] + '.jpg')


if __name__ == "__main__":
    main()
