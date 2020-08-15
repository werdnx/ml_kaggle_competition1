import pandas as pd


def main():
    test_df = pd.read_csv('/input/birdsong-recognition/test.csv')
    test_df.head()


if __name__ == "__main__":
    main()
