from tensorflow import keras
import time
from keras.models import load_model
import logging
import csv
import numpy as np
import cv2
from keras.optimizers import SGD, Adam
from efficientnet.keras import EfficientNetB3
import time

from python.model_params import target_size_, model_name

TARGET_ROWS = target_size_
TARGET_COLS = target_size_
CHANNELS = 3
DIR = '/input/jpeg/test512_nohair_croped/'


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
    X = np.zeros((1, TARGET_ROWS, TARGET_ROWS, CHANNELS), dtype=np.uint8)
    #print ('read img ' + DIR + img_label + '.jpg')
    img = read_image(DIR + img_label + '.jpg')
    # X[0, :] = img
    X[0, :] = cv2.resize(img, (TARGET_ROWS, TARGET_COLS), interpolation=cv2.INTER_CUBIC)
    return X


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
    model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    i = 0
    for image in images:
        test_set = prepare_data(image)
        x_test = test_set / 255
        result = model.predict(x_test)
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
        print (result)

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
