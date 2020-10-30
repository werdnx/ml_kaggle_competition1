import os
import sys

import librosa
import numpy as np
from tqdm import tqdm

from config import SAMPLE_RATE, TEST_PATH, PREPROCESS_PATH_TEST


def main():
    files_to_process = [(os.path.join(TEST_PATH, i), i) for i in os.listdir(sys.argv[1])]
    print("Audio preprocessing")
    for file in tqdm(files_to_process):
        if file[1].split(".")[0] != 'train_ground_truth':
            file_name = "{}".format(file[0].split(".")[0]) + '.wav'
            sound, r = librosa.load(file_name, sr=SAMPLE_RATE, mono=True)
            sound = librosa.util.normalize(sound, axis=0)
            np.save(PREPROCESS_PATH_TEST + file[1].split(".")[0] + '.npy', sound)


if __name__ == "__main__":
    main()
