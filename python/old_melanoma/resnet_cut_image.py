import os
import cv2
import numpy as np
from keras import applications
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, GlobalAveragePooling2D
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from preprare_train_data import split_data
from keras.optimizers import SGD, Adam
from tqdm import tqdm
from multiprocessing import Pool
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

ROWS = 256
COLS = 256
CHANNELS = 3
CLASSES = 2
DIR = '/input/jpeg/train256/'
DIR_POS = '/input/jpeg/train256_pos/'
CHUNK_SIZE = 1500


def read_image(file_path):
    return cv2.imread(file_path, cv2.IMREAD_COLOR)
    # return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in xrange(0, len(lst), n):
        yield lst[i:i + n]


def prepare_data_parallel(img_labels):
    pool = Pool(processes=4)
    X = []
    y = []
    results = pool.map(prepare_data, img_labels, len(img_labels) / 4)
    for res in results:
        X = np.concatenate((X, res[0]))
        y = np.concatenate((y, res[1]))
    return X, y


def additional_positive_cases():
    additional_len = 0
    train_images_pos = [(DIR_POS + i, i) for i in os.listdir(DIR_POS)]
    if len(train_images_pos) > 0:
        additional_len = len(train_images_pos)

    logging.info("additional len % s", additional_len)
    X = np.zeros((additional_len, ROWS, COLS, CHANNELS), dtype=np.uint8)
    y = np.zeros((1, additional_len), dtype=np.uint8)
    count = 0
    if len(train_images_pos) > 0:
        for i, image_file in enumerate(train_images_pos):
            X[count, :] = read_image(image_file[0])
            y[0, count] = 1
            count = count + 1
    return X, y


def prepare_data(img_labels):
    m = len(img_labels)
    logging.info("start processing of %s records", m)
    X = np.zeros((m, ROWS, COLS, CHANNELS), dtype=np.uint8)
    y = np.zeros((1, m), dtype=np.uint8)
    count = 0
    for i, item in enumerate(img_labels):
        X[count, :] = read_image(DIR + item[0] + '.jpg')
        if item[1] == 1:
            y[0, count] = 1
        else:
            y[0, count] = 0
        count = count + 1
        if count % 100 == 0:
            logging.info("process load data, iter = %s", count)
    return X, y


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def run():
    logging.info("start split_data")
    train, test = split_data()
    logging.info("train count = %s", len(train))
    logging.info("test count = %s", len(test))
    logging.info("start prepare train")

    base_model = applications.resnet50.ResNet50(weights=None, include_top=False, input_shape=(ROWS, COLS, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.7)(x)
    predictions = Dense(CLASSES, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    # sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    add_train_set_x, add_train_set_y = additional_positive_cases()
    train_chunks = chunks(train, CHUNK_SIZE)
    add_x_chunks = np.array_split(add_train_set_x, len(train_chunks))
    add_y_chunks = np.array_split(add_train_set_y, len(train_chunks))
    for train_chunk in train_chunks:
        train_set_x, train_set_y = prepare_data(train_chunk)
        fit_iteration(model, train_set_x, train_set_y)

    #logging.info("fit additional positive cases")
    #train_set_x, train_set_y = additional_positive_cases()
    #fit_iteration(model,train_set_x,train_set_y)

    logging.info("start prepare test")
    test_set_x, test_set_y = prepare_data(test)
    X_test = test_set_x / 255
    logging.info("start reshape y test")
    Y_test = convert_to_one_hot(test_set_y, CLASSES).T
    logging.info("number of test examples = %s", X_test.shape[0])
    logging.info("X_test shape: %s", X_test.shape)
    logging.info("Y_test shape: %s", Y_test.shape)

    preds = model.evaluate(X_test, Y_test)
    logging.info("Loss = %s" + str(preds[0]))
    logging.info("Test Accuracy = %s" + str(preds[1]))
    model.save('/input/model1_50rsnet_256_256_10_epoch3')
    logging.info("model saved")
    logging.info("preds ")
    for x in range(len(preds)):
        print (preds[x])
    model.summary()


def fit_iteration(model, train_set_x, train_set_y):
    X_train = train_set_x / 255
    logging.info("start reshape y train")
    Y_train = convert_to_one_hot(train_set_y, CLASSES).T
    logging.info("number of training examples = %s", X_train.shape[0])
    logging.info("X_train shape: %s", X_train.shape)
    logging.info("Y_train shape: %s", Y_train.shape)
    model.fit(X_train, Y_train, epochs=3, batch_size=8)
