import csv
import numpy as np
from sklearn.model_selection import train_test_split
from random import shuffle


def read_csv(path):
    positive = []
    negative = []
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print('Column names are ', ", ".join(row))
                line_count += 1
            else:
                if row[7] == 0:
                    negative.append((row[0], row[7]))
                else:
                    positive.append((row[0], row[7]))
                line_count += 1
        print('Processed lines.', line_count)
        return positive, negative


def split_data():
    pos, neg = read_csv("/input/train.csv")
    train_pos, test_pos = train_test_split(pos, test_size=0.33, random_state=42)
    train_neg, test_neg = train_test_split(neg, test_size=0.33, random_state=42)
    train = train_neg + train_pos
    shuffle(train)
    test = test_neg + test_pos
    shuffle(test)
    return train, test
