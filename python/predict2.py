from tensorflow import keras
import time
from keras.models import load_model
import logging
import csv
import numpy as np
import cv2
from keras.optimizers import SGD, Adam
from tensorflow.keras.applications import EfficientNetB6
import time
import tensorflow as tf
from utils import normalize_image_tf

from model_params import target_size_, model_name

TARGET_ROWS = target_size_
TARGET_COLS = target_size_
CHANNELS = 3
DIR = '/input/jpeg/test512_nohair/'


def read_data(path):
    result = []
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print('Column names are ', ", ".join(row))
                line_count += 1
            else:
                result.append(row[0])
                line_count += 1
        print('Processed lines.', line_count)
        return result


def normalize(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(target_size_, target_size_))
    image[:, :, 0] = clahe.apply(image[:, :, 0])
    return image


def prepare_data(img_label):
    print('load img ' + DIR + img_label + '.jpg')
    img = keras.preprocessing.image.load_img(
        DIR + img_label + '.jpg', target_size=(target_size_, target_size_)
    )
    # img = normalize_image_tf(img)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    return img_array


def read_image(file_path):
    return cv2.imread(file_path, cv2.IMREAD_COLOR)


def main():
    start = time.time()
    fields = ['image_name', 'target']
    rows = []
    stat_rows = []
    print("--------------------------------Run main!------------------------------------")
    images = read_data("/input/test.csv")
    print('loaded images')
    print(images)
    # time.sleep(10)
    # adam = Adam(lr=0.0001)
    model = load_model('/output/' + model_name)
    model.summary()
    # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    # model.compile(
    #     optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    # )
    i = 0
    for image in images:
        img_array = prepare_data(image)
        # ????
        result = model.predict(img_array)
        print (result)
        stat_rows.append((image, result[0][0], result[0][1]))
        # if result[0][1] > 0.4:
        #     rows.append((image, '1'))
        #     print('1')
        # else:
        #     rows.append((image, '0'))
        #     print('0')
        # #rows.append((image, np.argmax(result)))
        # i = i + 1
        # print('result1 ' + str(i))

    # exit(0)
    filename_stat = '/output/result_stat_' + model_name + '.csv'
    with open(filename_stat, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(stat_rows)

    # filename = '/output/result' + iteration+'.csv'
    # # writing to csv file
    # with open(filename, 'w') as csvfile:
    #     # creating a csv writer object
    #     csvwriter = csv.writer(csvfile)
    #     # writing the fields
    #     csvwriter.writerow(fields)
    #     # writing the data rows
    #     csvwriter.writerows(rows)
    print("--------------------------------End main!------------------------------------")
    end = time.time()
    total = end - start
    print("total time mls:")
    print(total)


if __name__ == "__main__":
    main()
