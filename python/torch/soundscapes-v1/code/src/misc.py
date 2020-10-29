import pandas as pd


def main():
    df = pd.read_csv(
        '/Users/dmitrenkoandrey/PycharmProjects/ml_kaggle_competition1/resources/soundscapes/train_ground_truth.csv',
        names=['id', 'class'])
    df.head()


if __name__ == "__main__":
    main()
