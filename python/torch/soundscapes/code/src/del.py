from sklearn.metrics import roc_auc_score


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
    # a = torch.tensor([[1, 2], [3, 4]])
    # b = torch.tensor([[5, 6], [7, 8]])
    # a = a[np.newaxis, ...][None, ...]
    # b = b[np.newaxis, ...][None, ...]
    # print(a)
    # print(b)
    # c = torch.cat((a, b), dim=0)
    # print('0')
    # print(c)
    #
    # c = torch.cat((a, b), dim=1)
    # print('1')
    # print(c)
    #
    # c = torch.cat((a, b), dim=2)
    # print('2')
    # print(c)
    # y = [7, 5, 3, 7, 1, 8, 5, 2, 0, 2, 8, 7, 0, 2, 0, 2]
    # preds = []
    # preds.append([0.09417012, 0.17261738, 0.10107411, 0.10563248, 0.0835258, 0.12026682,
    #               0.11351456, 0.11885612, 0.09034262])
    # preds.append([0.1024975, 0.11842676, 0.10726438, 0.08987355, 0.12457556, 0.1362249,
    #               0.13359985, 0.09284353, 0.09469399])
    # preds.append([0.10669417, 0.080603, 0.12004627, 0.0882355, 0.07650611, 0.09960859,
    #               0.16327615, 0.15004021, 0.11499003])
    # preds.append([0.08789872, 0.11311763, 0.12273856, 0.11800566, 0.12020522, 0.10852592,
    #               0.12222297, 0.1189508, 0.08833445])
    # preds.append([0.10088749, 0.09179572, 0.09137488, 0.14264572, 0.12208412, 0.09983024,
    #               0.09218597, 0.1463339, 0.11286198])
    # preds.append([0.08364383, 0.10794599, 0.13029446, 0.07123296, 0.11326433, 0.08445083,
    #               0.126917, 0.16300923, 0.11924128])
    # preds.append([0.10001411, 0.09919977, 0.10415963, 0.09310031, 0.10509419, 0.14543393,
    #               0.11063023, 0.08780488, 0.15456292])
    # preds.append([0.12889397, 0.12896648, 0.1135973, 0.10372277, 0.11544419, 0.10548102,
    #               0.10027764, 0.10516422, 0.09845238])
    # preds.append([0.13302362, 0.14364304, 0.1224416, 0.10367337, 0.11041989, 0.12382857,
    #               0.08436997, 0.09283999, 0.08575995])
    # preds.append([0.06062314, 0.10088311, 0.1216234, 0.09195877, 0.1158341, 0.12972659,
    #               0.11073206, 0.11461497, 0.15400386])
    # preds.append([0.11206343, 0.09950636, 0.09829276, 0.1007413, 0.09682902, 0.08810791,
    #               0.11026044, 0.14286752, 0.15133129])
    # preds.append([0.0946971, 0.13084424, 0.10578493, 0.0874607, 0.08663909, 0.14362267,
    #               0.1308988, 0.12021964, 0.09983291])
    # preds.append([0.07609089, 0.09753932, 0.06911447, 0.09458329, 0.11094119, 0.16617242,
    #               0.16768876, 0.10005361, 0.11781598])
    # preds.append([0.12615865, 0.09542786, 0.07302707, 0.09621997, 0.12524566, 0.11731122,
    #               0.11136972, 0.15950757, 0.0957323])
    # preds.append([0.0965052, 0.10337518, 0.08751525, 0.11467645, 0.11594169, 0.12232672,
    #               0.13512236, 0.1017479, 0.12278908])
    # preds.append([0.07774663, 0.13928312, 0.08461349, 0.14408943, 0.1126513, 0.1031479,
    #               0.0959421, 0.11304583, 0.1294802])
    # loss = roc_auc_score(y, preds, multi_class="ovr", labels=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    # print(loss)
    for i in range(10):
        print(i)


if __name__ == "__main__":
    main()
