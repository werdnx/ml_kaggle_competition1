import pandas as pd

base_path = '/Users/dmitrenkoandrey/PycharmProjects/ml_kaggle_competition1/result/'
preds = [
    ['submission_b6-512-model_4+4+4_epoch.csv', 0.4],
    ['submission_features_preds.csv', 0.6],
    # ['submission2_4fold_efnet_5_iter1__fold-1-512x512.model.csv', 0.15],
    # ['submission2_4fold_efnet_4_fold-1-384x384.model.csv', 0.15]
    # ['submission2_4fold_efnet_0_fold-0-128x128.model.csv', 0.15]
    #      'submission2_4fold_efnet_1_fold-0-192x192.model.csv',
    #        'submission2_4fold_efnet_2_fold-0-256x256.model.csv',

]


def main():
    pred_b6 = pd.read_csv(base_path + preds[0][0])
    pred_meta = pd.read_csv(base_path + preds[1][0])
    # pred_b5 = pd.read_csv(base_path + preds[2][0])
    # pred3 = pd.read_csv(base_path + preds[3][0])
    # pred4 = pd.read_csv(base_path + preds[4])
    # pred5 = pd.read_csv(base_path + preds[5])
    # pred6 = pd.read_csv(base_path + preds[6])
    sample = pd.read_csv(base_path + 'sample_submission.csv')
    print('prob of one = ' + str(1.0 / len(preds)))

    # pred6['target_1'] = 1.0 - pred6['target_0']
    sample['target'] = (

            pred_b6['target'] * preds[0][1] +
            pred_meta['target'] * preds[1][1]
            # pred_b5['target'] * preds[2][1] +
            # pred3['target'] * preds[3][1]
        # pred4['target'] * 1.0 / len(preds) +
        # pred5['target'] * 1.0 / len(preds)
        # + pred6['target'] * 1.0 / len(preds)

    )
    sample.to_csv(base_path + 'ensembled_b6_40_meta.csv', header=True, index=False)


if __name__ == "__main__":
    main()
