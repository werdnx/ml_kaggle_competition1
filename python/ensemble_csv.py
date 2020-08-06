import pandas as pd

base_path = '/Users/dmitrenkoandrey/PycharmProjects/ml_kaggle_competition1/result/'
preds = ['submission_128.csv',
         'submission_192.csv',
         'submission_256.csv'
         ]


def main():
    pred0 = pd.read_csv(base_path + preds[0])
    pred1 = pd.read_csv(base_path + preds[1])
    pred2 = pd.read_csv(base_path + preds[2])
    # pred3 = pd.read_csv(base_path + preds[3])
    # pred4 = pd.read_csv(base_path + preds[4])
    # pred5 = pd.read_csv(base_path + preds[5])
    # pred6 = pd.read_csv(base_path + preds[6])
    sample = pd.read_csv('/Users/dmitrenkoandrey/PycharmProjects/ml_kaggle_competition1/result/sample_submission.csv')
    print('prob of one = ' + str(1.0 / len(preds)))

    # pred6['target_1'] = 1.0 - pred6['target_0']
    sample['target'] = (

        pred0['target'] * 1.0 / len(preds) +
        pred1['target'] * 1.0 / len(preds) +
        pred2['target'] * 1.0 / len(preds)
        # pred3['target'] * 1.0 / len(preds) +
        # pred4['target'] * 1.0 / len(preds) +
        # pred5['target'] * 1.0 / len(preds)
        #
        # + pred6['target_0'] * 1.0 / len(preds)

    )
    # sample.drop('target', axis=1, inplace=True)
    # sample['target_0'] = (
    #
    #     pred0['target_0'] * 1.0 / len(preds) +
    #     pred1['target_0'] * 1.0 / len(preds) +
    #     pred2['target_0'] * 1.0 / len(preds) +
    #     pred3['target_0'] * 1.0 / len(preds) +
    #     pred4['target_0'] * 1.0 / len(preds) +
    #     pred5['target_0'] * 1.0 / len(preds)
    #
    #     + pred6['target_0'] * 1.0 / len(preds)
    #
    # )
    #
    # sample['target_1'] = (
    #
    #     pred0['target_1'] * 1.0 / len(preds) +
    #     pred1['target_1'] * 1.0 / len(preds) +
    #     pred2['target_1'] * 1.0 / len(preds) +
    #     pred3['target_1'] * 1.0 / len(preds) +
    #     pred4['target_1'] * 1.0 / len(preds) +
    #     pred5['target_1'] * 1.0 / len(preds)
    #
    #     + pred6['target_1'] * 1.0 / len(preds)
    #
    # )

    sample.to_csv(base_path + 'ensembled_with_features.csv', header=True, index=False)


if __name__ == "__main__":
    main()
