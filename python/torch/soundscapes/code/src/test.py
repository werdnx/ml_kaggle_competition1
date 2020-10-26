import os
import sys

import pandas as pd


def test(data_folder, submission_path):
    files_to_predict = [(os.path.join(data_folder, i), i) for i in os.listdir(data_folder)]
    result = []
    for file in files_to_predict:
        result.append(
            {"id": "{}".format(file[1].split(".")[0]), "A": 0.5, "B": 0.5, "C": 0.5, "D": 0.5, "E": 0.5, "F": 0.5,
             "G": 0.5, "H": 0.5, "I": 0.5})
    result_df = pd.DataFrame(result)
    result_df.to_csv(submission_path, header=True, index=False)


def main():
    test(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
