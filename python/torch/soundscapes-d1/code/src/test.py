import os
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import TEST_PATH, FOLDS, MODEL_PATH, PREPROCESS_PATH_TEST
from sound_dataset import SoundDatasetTest
from utils import process_sound

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


def prepare_predict_data(files_to_predict):
    result = []
    for file in tqdm(files_to_predict):
        if file[1].split(".")[0] != 'train_ground_truth':
            file_name = PREPROCESS_PATH_TEST + file[1].split(".")[0] + '.npy'
            sound = np.load(file_name)
            to_predict = process_sound(sound, False)
            result.append({0: file[1], 1: to_predict})
    return result


def test(data_folder, submission_path):
    files_to_predict = [(os.path.join(TEST_PATH, i), i) for i in os.listdir(data_folder)]
    dfs = []
    # predict_data = prepare_predict_data(files_to_predict)
    # sm = torch.nn.Softmax(dim=1)
    train_set = SoundDatasetTest(data_folder)
    test_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False, num_workers=4)
    for fold in tqdm(np.arange(FOLDS)):
        print('process fold ' + str(fold))
        model = torch.load(MODEL_PATH + '_fold' + str(fold))
        model.eval()
        result = []
        for data, target in tqdm(test_loader):
            if data is not None:
                data = data.half()
                data = data.to(device)
                output = model(data)
                arr = torch.exp(output).data.cpu().numpy()
                # file_name = "{}".format(file[0].split(".")[0]) + '.wav'
                # to_predict = process_file(file_name, False)
                # clip_np = spec_to_image(get_melspectrogram_db(file_name))[np.newaxis, ...]
                # output = model(torch.from_numpy(clip_np).float()[None, ...].to(device))
                # output = model(p_data[1][None, ...].to(device))
                # arr = sm(output).data.cpu().numpy()
                # arr = torch.exp(output).data.cpu().numpy()
                # arr = output.data.cpu().numpy()
                # print('arr is ')
                # print(arr)
                # arr = arr[0]
                result.append(
                    {"id": "{}".format(target[0]), "A": arr[0][0], "B": arr[0][1],
                     "C": arr[0][2], "D": arr[0][3],
                     "E": arr[0][4],
                     "F": arr[0][5], "G": arr[0][6], "H": arr[0][7],
                     "I": arr[0][8]})
            else:
                print("skip train_ground_truth")
        dfs.append(pd.DataFrame(result))

    # result_df = pd.DataFrame(result)
    result_df = dfs[0].copy()
    result_df['A'] = 0.0
    result_df['B'] = 0.0
    result_df['C'] = 0.0
    result_df['D'] = 0.0
    result_df['E'] = 0.0
    result_df['F'] = 0.0
    result_df['G'] = 0.0
    result_df['H'] = 0.0
    result_df['I'] = 0.0
    for df in dfs:
        result_df['A'] = result_df['A'] + df['A']
        result_df['B'] = result_df['B'] + df['B']
        result_df['C'] = result_df['C'] + df['C']
        result_df['D'] = result_df['D'] + df['D']
        result_df['E'] = result_df['E'] + df['E']
        result_df['F'] = result_df['F'] + df['F']
        result_df['G'] = result_df['G'] + df['G']
        result_df['H'] = result_df['H'] + df['H']
        result_df['I'] = result_df['I'] + df['I']

    result_df['A'] = result_df['A'] / len(dfs)
    result_df['B'] = result_df['B'] / len(dfs)
    result_df['C'] = result_df['C'] / len(dfs)
    result_df['D'] = result_df['D'] / len(dfs)
    result_df['E'] = result_df['E'] / len(dfs)
    result_df['F'] = result_df['F'] / len(dfs)
    result_df['G'] = result_df['G'] / len(dfs)
    result_df['H'] = result_df['H'] / len(dfs)
    result_df['I'] = result_df['I'] / len(dfs)
    print('res df norm')
    print(result_df.head())
    result_df.to_csv(submission_path, header=True, index=False, float_format='%.2f')


def main():
    test(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
