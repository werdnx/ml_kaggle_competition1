import os
import cv2
import numpy as np
#from os import cpu_count
#from multiprocessing.pool import Pool
#from functools import partial
from tqdm import tqdm

IN_DIR = '/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1/input/jpeg/test/'
OUT_DIR = '/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1/input/jpeg/test512/'
ROWS = 512
COLS = 512


def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)


def prepare_img(images):
    l = len(images)
    with tqdm(total=100) as pbar:
        for i, image_file in enumerate(images):
            img = read_image(image_file[0])
            cv2.imwrite(OUT_DIR+image_file[1], img)
            pbar.update((i * 100) / float(l))


def main():
    train_images_all = [(IN_DIR + i, i) for i in os.listdir(IN_DIR)]
    train_images_512 = [(IN_DIR + i, i) for i in os.listdir(OUT_DIR)]
    to_proccess = list(set(train_images_all) - set(train_images_512))
    prepare_img(to_proccess)
    #four_split = np.array_split(train_images, 4)
    # with Pool(processes=cpu_count()) as p:
    #for array in train_images:
        #prepare_img(array)
        #p.imap_unordered(prepare_img, array)







if __name__ == "__main__":
    main()