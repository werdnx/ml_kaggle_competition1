import os
import sys

import pandas as pd


def train(data_folder):
    df = pd.read_csv(os.path.join(data_folder, 'train_ground_truth.csv'))


def main():
    train(sys.argv[1])


if __name__ == "__main__":
    main()
