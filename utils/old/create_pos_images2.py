import os
import cv2
import csv
import numpy as np

# from os import cpu_count
# from multiprocessing.pool import Pool
# from functools import partial

IN_DIR = '/home/werdn/input/jpeg/train_nohair/'
OUT_DIR = '/home/werdn/input/jpeg/tsds/train/malignant/'


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


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def read_image_and_rotate(file_path, angle):
    print ("read file " + file_path)
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    height, width, channels = img.shape
    return rotate_image(img, angle)


def prepare_img(images, ind):
    for i, image_file in enumerate(images):
        img = cv2.imread(IN_DIR + image_file + '.jpg', cv2.IMREAD_COLOR)
        print ("write file " + OUT_DIR + image_file + '_' + str(ind) + '.jpg')
        cv2.imwrite(OUT_DIR + image_file + '_' + str(ind) + '.jpg', img)


def main():
    pos_images = read_csv("/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1/input/train.csv")
    for i in range(0, 60):
        prepare_img(pos_images, i)
    # four_split = np.array_split(train_images, 4)
    # with Pool(processes=cpu_count()) as p:
    # for array in train_images:
    # prepare_img(array)
    # p.imap_unordered(prepare_img, array)


if __name__ == "__main__":
    main()
