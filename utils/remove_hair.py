import os
import numpy as np
import cv2
import multiprocessing as mp
from pathlib import Path

IN_DIR = '/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1/input/jpeg/train/'
OUT_DIR = '/home/werdn/input/jpeg/train_nohair/'


def hair_remove(image):
    # convert image to grayScale
    grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # kernel for morphologyEx
    kernel = cv2.getStructuringElement(1, (17, 17))
    # apply MORPH_BLACKHAT to grayScale image
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    # apply thresholding to blackhat
    _, threshold = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    # inpaint with original image and threshold image
    final_image = cv2.inpaint(image, threshold, 1, cv2.INPAINT_TELEA)

    return final_image


def process_one_img(image_file):
    my_file = Path(OUT_DIR + image_file[1])
    if my_file.is_file():
        # file exists
        print('file exists ' + image_file[1])
        return
    else:
        print('start process file ' + image_file[1])
        img = cv2.imread(image_file[0], cv2.IMREAD_COLOR)
        no_hair_img = hair_remove(img)
        cv2.imwrite(OUT_DIR + image_file[1], no_hair_img)
        print('processed file ' + image_file[1])


def main():
    train_images_512 = [(IN_DIR + i, i) for i in os.listdir(IN_DIR)]
    for i, image_file in enumerate(train_images_512):
        process_one_img(image_file)
    # imgs = [(IN_DIR + i, i) for i in os.listdir(IN_DIR)]
    # pool = mp.Pool(processes=4)
    # outputs = pool.map(process_one_img, imgs)
    # pool.close()


if __name__ == "__main__":
    main()
