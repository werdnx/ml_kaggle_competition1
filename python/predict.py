from tensorflow import keras
import time
from keras.models import load_model
import logging
import csv
import numpy as np
import cv2
from keras.optimizers import SGD, Adam
from efficientnet.keras import EfficientNetB3

ROWS = 512
COLS = 512
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


def prepare_data(img_label):
    X = np.zeros((1, ROWS, COLS, CHANNELS), dtype=np.uint8)
    X[0, :] = read_image(DIR + img_label + '.jpg')
    return X


def read_image(file_path):
    return cv2.imread(file_path, cv2.IMREAD_COLOR)


def main():
    start = time.time()
    fields = ['image_name', 'target']
    rows = []
    print("--------------------------------Run main!------------------------------------")
    images = read_data("/input/test.csv")
    #adam = Adam(lr=0.0001)
    model = load_model('/output/model1_EfficientNetB3_gen2')
    #model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    i = 0
    for image in images:
        test_set = prepare_data(image)
        x_test = test_set / 255
        result = model.predict(x_test)
        rows.append((image, np.argmax(result)))
        i = i + 1
        logging("result for image %s is %s, num = %s", image, np.argmax(result), str(i))

    #exit(0)
    filename = "/output/result4.csv"
    # writing to csv file
    with open(filename, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        # writing the fields
        csvwriter.writerow(fields)
        # writing the data rows
        csvwriter.writerows(rows)
    print("--------------------------------End main!------------------------------------")
    end = time.time()
    total = end - start
    print("total time mls:")
    print(total)


if __name__ == "__main__":
    main()
