import torch
import numpy as np


def main():
    # probs = np.zeros(5)
    # a1 = [0.2, 0.3, 0.4, 0.1, 0.2]
    # a2 = [0.2, 0.3, 0.4, 0.1, 0.2]
    # probs = np.add(probs, a1)
    # probs = np.add(probs, a2)
    # probs = probs / 2.0
    # print(probs)
    # print(np.mean(probs))
    # dfs = []
    # for f in np.arange(5):
    #     result = []
    #     for i in np.arange(10):
    #         result.append(
    #             {"id": i, "A": i, "B": i + 1})
    #     dfs.append(pd.DataFrame(result))
    #
    # result_df = dfs[0].copy()
    # result_df['A'] = 0.0
    # result_df['B'] = 0.0
    # for df in dfs:
    #     print("df")
    #     print(df.head())
    #     result_df['A'] = result_df['A'] + df['A']
    #     result_df['B'] = result_df['B'] + df['B']
    #
    # print('res df')
    # print(result_df.head())
    # result_df['A'] = result_df['A'] / len(dfs)
    # result_df['B'] = result_df['B'] / len(dfs)
    # print('res df norm')
    # print(result_df.head())
    #
    # skf = KFold(n_splits=5, shuffle=True, random_state=42)
    # for fold, (idxT, idxV) in enumerate(skf.split(np.arange(2300))):
    #     print('Train')
    #     print(idxT)
    #     print('Validation')
    #     print(idxV)
    # msk = np.random.rand(2300) < 0.7
    # print(msk)
    # f = process_file('/Users/dmitrenkoandrey/PycharmProjects/ml_kaggle_competition1/temp/1765711516.wav')
    # print(f)
    # f = spec_to_image(get_melspectrogram_db(file_path))
    # sound, r = librosa.load('/Users/dmitrenkoandrey/PycharmProjects/ml_kaggle_competition1/temp/1765711516.wav',
    #                         sr=16000, mono=True)
    # sound = librosa.util.normalize(sound, axis=0)
    # np.save('/Users/dmitrenkoandrey/PycharmProjects/ml_kaggle_competition1/temp/s1.npy',sound)
    # f1 = get_one_sample_from_file('/Users/dmitrenkoandrey/PycharmProjects/ml_kaggle_competition1/temp/s1.npy')
    # f2 = get_samples_from_file('/Users/dmitrenkoandrey/PycharmProjects/ml_kaggle_competition1/temp/s1.npy')
    # print('done')
    # metadata = audio_metadata.load('/Users/dmitrenkoandrey/PycharmProjects/ml_kaggle_competition1/temp/0000322837.flac')
    # print(metadata.streaminfo.md5)
    a = torch.tensor([[1, 2], [3, 4]])
    b = torch.tensor([[5, 6], [7, 8]])
    a = a[np.newaxis, ...][None, ...]
    b = b[np.newaxis, ...][None, ...]
    print(a)
    print(b)
    c = torch.cat((a, b), dim=0)
    print('0')
    print(c)

    c = torch.cat((a, b), dim=1)
    print('1')
    print(c)

    c = torch.cat((a, b), dim=2)
    print('2')
    print(c)


if __name__ == "__main__":
    main()
