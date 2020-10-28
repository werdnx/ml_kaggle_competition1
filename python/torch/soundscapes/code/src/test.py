import os
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from audioutils import spec_to_image, get_melspectrogram_db
from train import MODEL_PATH

# TODO REPLACE BY TEST
TEST_PATH = '/wdata/test'
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


def test(data_folder, submission_path):
    model = torch.load(MODEL_PATH)
    model.eval()
    files_to_predict = [(os.path.join(TEST_PATH, i), i) for i in os.listdir(data_folder)]
    result = []
    sm = torch.nn.Softmax(dim=1)
    for file in tqdm(files_to_predict):
        if file[1].split(".")[0] != 'train_ground_truth':
            file_name = "{}".format(file[0].split(".")[0]) + '.wav'
            clip_np = spec_to_image(get_melspectrogram_db(file_name))[np.newaxis, ...]
            output = model(torch.from_numpy(clip_np).float()[None, ...].to(device))
            arr = sm(output).data.cpu().numpy()
            result.append(
                {"id": "{}".format(file[1].split(".")[0]), "A": arr[0][0], "B": arr[0][1],
                 "C": arr[0][2], "D": arr[0][3],
                 "E": arr[0][4],
                 "F": arr[0][5], "G": arr[0][6], "H": arr[0][7],
                 "I": arr[0][8]})
        else:
            print("skip train_ground_truth")

    result_df = pd.DataFrame(result)
    result_df.to_csv(submission_path, header=True, index=False, float_format='%.2f')


def main():
    test(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
