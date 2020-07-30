import cv2
import numpy as np
import albumentations as A
import enum
import os
from augmentation import size
from augmentation import gaussianBlur
from augmentation import imageSegmentation2

IN_DIR = '/home/werdn/input/jpeg/tsds/train/benign/'
OUT_DIR = '/home/werdn/input/jpeg/tsds/train/benign/'

func_list = [gaussianBlur, imageSegmentation2]


def aug_iteration(image_file):
    img = cv2.imread(image_file[0], cv2.IMREAD_COLOR)
    img = cv2.resize(img, (size, size))
    for aug in func_list:
        out_path = OUT_DIR + aug.__name__ + '_' + image_file[1]
        print ("write file " + out_path)
        aug(img, out_path)

def main():
    test_images = [(IN_DIR + i, i) for i in os.listdir(IN_DIR)]
    for i, image_file in enumerate(test_images):
        aug_iteration(image_file)


if __name__ == "__main__":
    main()
