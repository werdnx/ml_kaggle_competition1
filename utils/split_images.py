import os
import cv2
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from shutil import copyfile
import cv2
import random
from augmentation import size

# from os import cpu_count
# from multiprocessing.pool import Pool
# from functools import partial
IN_DIR = '/home/werdn/input/jpeg/train512_nohair/'
IN_DIR_TEST = '/home/werdn/input/jpeg/test512_nohair/'
OUT_DIR_TEST = '/home/werdn/input/jpeg/test512_nohair_normalized/'
OUT_DIR = '/home/werdn/input/jpeg/tsds/'
IN_DIR_MAL = '/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1/output/malign_candidates/'


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
    train, test = train_test_split(train_df, test_size=0.2)

    mal_images = [(IN_DIR_MAL + i, i) for i in os.listdir(IN_DIR_MAL)]
    random.shuffle(mal_images)
    length = len(mal_images)
    test_index = length // 5
    test_mal = mal_images[:test_index]
    train_mal = mal_images[test_index:]
    print('len of add malign test ' + str(test_mal))
    print('len of add malign train ' + str(train_mal))
    for index, row in enumerate(train_mal):
        copyfile(IN_DIR_MAL + row[1], OUT_DIR + 'train/malignant/' + row[1])
        print ("write file " + OUT_DIR + 'train/malignant/' + row[1] )
    for index, row in enumerate(test_mal):
        copyfile(IN_DIR_MAL + row[1], OUT_DIR + 'test/malignant/' + row[1])
        print ("write file " + OUT_DIR + 'test/malignant/' + row[1] )

    for index, row in train.iterrows():
        if str(row['target']) == "1":
            # normalize(IN_DIR + row['image_name'] + '.jpg', OUT_DIR + 'train/malignant/' + row['image_name'] + '.jpg')
            copyfile(IN_DIR + row['image_name'] + '.jpg', OUT_DIR + 'train/malignant/' + row['image_name'] + '.jpg')
            print ("write file " + OUT_DIR + 'train/malignant/' + row['image_name'] + '.jpg')
        else:
            # normalize(IN_DIR + row['image_name'] + '.jpg', OUT_DIR + 'train/benign/' + row['image_name'] + '.jpg')
            copyfile(IN_DIR + row['image_name'] + '.jpg', OUT_DIR + 'train/benign/' + row['image_name'] + '.jpg')
            print ("write file " + OUT_DIR + 'train/benign/' + row['image_name'] + '.jpg')
    for index, row in test.iterrows():
        if str(row['target']) == "1":
            # normalize(IN_DIR + row['image_name'] + '.jpg', OUT_DIR + 'test/malignant/' + row['image_name'] + '.jpg')
            copyfile(IN_DIR + row['image_name'] + '.jpg', OUT_DIR + 'test/malignant/' + row['image_name'] + '.jpg')
            print ("write file " + OUT_DIR + 'test/malignant/' + row['image_name'] + '.jpg')
        else:
            # normalize(IN_DIR + row['image_name'] + '.jpg', OUT_DIR + 'test/benign/' + row['image_name'] + '.jpg')
            copyfile(IN_DIR + row['image_name'] + '.jpg', OUT_DIR + 'test/benign/' + row['image_name'] + '.jpg')
            print ("write file " + OUT_DIR + 'test/benign/' + row['image_name'] + '.jpg')


if __name__ == "__main__":
    main()
