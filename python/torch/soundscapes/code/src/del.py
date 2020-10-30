import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from utils import process_file


def main():
    dfs = []
    for f in np.arange(5):
        result = []
        for i in np.arange(10):
            result.append(
                {"id": i, "A": i, "B": i + 1})
        dfs.append(pd.DataFrame(result))

    result_df = dfs[0].copy()
    result_df['A'] = 0.0
    result_df['B'] = 0.0
    for df in dfs:
        print("df")
        print(df.head())
        result_df['A'] = result_df['A'] + df['A']
        result_df['B'] = result_df['B'] + df['B']

    print('res df')
    print(result_df.head())
    result_df['A'] = result_df['A'] / len(dfs)
    result_df['B'] = result_df['B'] / len(dfs)
    print('res df norm')
    print(result_df.head())


    skf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (idxT, idxV) in enumerate(skf.split(np.arange(2300))):
        print('Train')
        print(idxT)
        print('Validation')
        print(idxV)
    msk = np.random.rand(2300) < 0.7
    print(msk)
    f = process_file('/Users/dmitrenkoandrey/PycharmProjects/ml_kaggle_competition1/temp/1765711516.wav')
    print(f)



if __name__ == "__main__":
    main()
