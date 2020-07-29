import os
import cv2
import csv
import numpy as np

# from os import cpu_count
# from multiprocessing.pool import Pool
# from functools import partial

IN_DIR = '/input/jpeg/train512_nohair/'
OUT_DIR = '/input/jpeg/tsds/train/malignant/'
ROWS = 512
COLS = 512


def read_csv(path):
    positive = []
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print('Column names are ', ", ".join(row))
                line_count += 1
            else:
                if row[7] == "1":
                    positive.append(row[0])
                line_count += 1
        print('Processed lines.', line_count)
        print('positive lines.', positive)
        return positive


def main():
    to_augument_images = read_csv("/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1/input/train.csv")



if __name__ == "__main__":
    main()
