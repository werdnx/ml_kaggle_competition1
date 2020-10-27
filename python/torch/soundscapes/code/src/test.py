import os
import sys

import pandas as pd


def test(data_folder, submission_path):
    files_to_predict = [(os.path.join(data_folder, i), i) for i in os.listdir(data_folder)]
    result = []
    for file in files_to_predict:
        result.append(
            {"id": "{}".format(file[1].split(".")[0]), "A": 0.1, "B": 0.1, "C": 0.1, "D": 0.1, "E": 0.1, "F": 0.1,
             "G": 0.1, "H": 0.1, "I": 0.1})
    result_df = pd.DataFrame(result)
    result_df.to_csv(submission_path, header=True, index=False)


def main():
    test(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
