import math
import os
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from audioutils import get_samples_from_file
from config import TEST_PATH, MODEL_PATH, PREPROCESS_PATH_TEST, MODEL_PARAMS

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


def prepare_predict_data(files_to_predict, seconds):
    result = []
    for file in tqdm(files_to_predict):
        if file[1].split(".")[0] != 'train_ground_truth':
            file_name = PREPROCESS_PATH_TEST + file[1].split(".")[0] + '.npy'
            crops = get_samples_from_file(file_name, seconds)
            # crops = get_one_sample_from_file(file_name)
            result.append({0: file[1], 1: crops})
    return result


def test(data_folder, submission_path):
    files_to_predict = [(os.path.join(TEST_PATH, i), i) for i in os.listdir(data_folder)]
    dfs = []
    for model_param in MODEL_PARAMS:
        predict_data = prepare_predict_data(files_to_predict, model_param['SECONDS'])
        # sm = torch.nn.Softmax(dim=1)
        # train_set = SoundDatasetTest(data_folder)
        # test_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False, num_workers=4)
        # for fold in tqdm(np.arange(FOLDS)):
        print('process model ' + model_param['NAME'])
        model = torch.load(MODEL_PATH + '_' + model_param['NAME'])
        model.eval()
        result = []
        for crops_data in tqdm(predict_data):
            probs = np.zeros(9)
            for crop in crops_data[1]:
                crop = crop[np.newaxis, ...]
                # crop = crop[None, ...]
                crop = crop.half()
                crop = crop.to(device)
                output = model(crop)
                # arr = output.data.cpu().numpy()
                arr = torch.exp(output).data.cpu().numpy()
                # arr[0] = [round_decimals_down(x) for x in arr[0]]
                probs = np.add(probs, arr[0])
                # print('probs:')
                # print(probs)
            # print('probs ')
            # print(probs)
            probs = probs / float(len(crops_data[1]))
            probs = np.clip(probs, 0.01, 0.99)
            # probs = [min(x, 0.99) for x in probs]
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
                {"id": "{}".format(crops_data[0].split(".")[0]), "A": probs[0], "B": probs[1],
                 "C": probs[2], "D": probs[3],
                 "E": probs[4],
                 "F": probs[5], "G": probs[6], "H": probs[7],
                 "I": probs[8]})
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


def round_decimals_down(number: float, decimals: int = 2):
    """
    Returns a value rounded down to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.floor(number)

    factor = 10 ** decimals
    return math.floor(number * factor) / factor


def main():
    test(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
