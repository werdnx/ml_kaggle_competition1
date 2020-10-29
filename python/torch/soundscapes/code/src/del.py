import numpy as np
import pandas as pd


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


if __name__ == "__main__":
    main()
