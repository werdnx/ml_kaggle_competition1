import pandas as pd

base_path = '/result/'
preds = [
    # 'submission2_4fold_efnet_0_fold-0-128x128.model.csv',
    #      'submission2_4fold_efnet_1_fold-0-192x192.model.csv',
    #        'submission2_4fold_efnet_2_fold-0-256x256.model.csv',
    #      'submission2_4fold_efnet_4_fold-1-384x384.model.csv',
    'submission2_4fold_efnet_5_iter1__fold-1-512x512.model.csv',
    'submission_b6-512-model_4+4+4_epoch.csv',
    'submission_features_preds.csv'
]


def main():
    pred0 = pd.read_csv(base_path + preds[0])
    pred1 = pd.read_csv(base_path + preds[1])
    pred2 = pd.read_csv(base_path + preds[2])
    # pred3 = pd.read_csv(base_path + preds[3])
    # pred4 = pd.read_csv(base_path + preds[4])
    # pred5 = pd.read_csv(base_path + preds[5])
    # pred6 = pd.read_csv(base_path + preds[6])
    sample = pd.read_csv('/result/sample_submission.csv')
    print('prob of one = ' + str(1.0 / len(preds)))

    # pred6['target_1'] = 1.0 - pred6['target_0']
    sample['target'] = (

            pred0['target'] * 1.0 / len(preds) +
            pred1['target'] * 1.0 / len(preds)
            + pred2['target'] * 1.0 / len(preds)
        # + pred3['target'] * 1.0 / len(preds) +
        # pred4['target'] * 1.0 / len(preds) +
        # pred5['target'] * 1.0 / len(preds)
        # + pred6['target'] * 1.0 / len(preds)

    )
    sample.to_csv(base_path + 'ensembled_b6_b6_with_meta.csv', header=True, index=False)


if __name__ == "__main__":
    main()
