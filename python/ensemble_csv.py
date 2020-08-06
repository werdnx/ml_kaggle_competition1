import pandas as pd

base_path = 'D:\\pycharm\\frd\\resources\\result_stat\\'
preds = ['result_stat_model1_EfficientNetB5_gen_456_10_30_11.csv',
         'result_stat_model3_EfficientNetB3_gen_300_15_15_12.csv',
         'result_stat_model3_EfficientNetB3_gen_300_15_15_13.csv',
         'result_stat_model4_EfficientNetB5_gen_456_6_15_14.csv',
         'result_stat_model5_EfficientNetB5_gen_456_6_15_15.csv',
         'result_stat_model7_EfficientNetB3B4B5_gen_380_10_12_17.csv'
         ,'features_preds.csv'
         ]


def main():
    pred0 = pd.read_csv(base_path + preds[0])
    pred1 = pd.read_csv(base_path + preds[1])
    pred2 = pd.read_csv(base_path + preds[2])
    pred3 = pd.read_csv(base_path + preds[3])
    pred4 = pd.read_csv(base_path + preds[4])
    pred5 = pd.read_csv(base_path + preds[5])
    pred6 = pd.read_csv(base_path + preds[6])
    sample = pd.read_csv('D:\pycharm\\frd\\resources\\add_features\\sample_submission.csv')
    print('prob of one = ' + str(1.0 / len(preds)))

    pred6['target_1'] = 1.0 - pred6['target_0']

    sample.drop('target', axis=1, inplace=True)
    sample['target_0'] = (

        pred0['target_0'] * 1.0 / len(preds) +
        pred1['target_0'] * 1.0 / len(preds) +
        pred2['target_0'] * 1.0 / len(preds) +
        pred3['target_0'] * 1.0 / len(preds) +
        pred4['target_0'] * 1.0 / len(preds) +
        pred5['target_0'] * 1.0 / len(preds)

        + pred6['target_0'] * 1.0 / len(preds)

    )

    sample['target_1'] = (

        pred0['target_1'] * 1.0 / len(preds) +
        pred1['target_1'] * 1.0 / len(preds) +
        pred2['target_1'] * 1.0 / len(preds) +
        pred3['target_1'] * 1.0 / len(preds) +
        pred4['target_1'] * 1.0 / len(preds) +
        pred5['target_1'] * 1.0 / len(preds)

        + pred6['target_1'] * 1.0 / len(preds)

    )

    sample.to_csv('D:\\pycharm\\frd\\resources\\result\\ensembled_with_features.csv', header=True, index=False)


if __name__ == "__main__":
    main()
