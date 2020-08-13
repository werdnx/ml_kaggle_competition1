import pandas as pd
import xgboost as xgb

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, cross_validate
from sklearn.metrics import roc_auc_score, roc_curve


def model_check(X_train, y_train, estimators, cv):
    model_table = pd.DataFrame()

    row_index = 0
    for est in estimators:
        MLA_name = est.__class__.__name__
        model_table.loc[row_index, 'Model Name'] = MLA_name

        cv_results = cross_validate(est,
                                    X_train,
                                    y_train,
                                    cv=cv,
                                    scoring='roc_auc',
                                    return_train_score=True,
                                    n_jobs=-1)

        model_table.loc[row_index,
                        'Train roc Mean'] = cv_results['train_score'].mean()
        model_table.loc[row_index,
                        'Test roc Mean'] = cv_results['test_score'].mean()
        model_table.loc[row_index, 'Test Std'] = cv_results['test_score'].std()
        model_table.loc[row_index, 'Time'] = cv_results['fit_time'].mean()

        row_index += 1

    model_table.sort_values(by=['Test roc Mean'],
                            ascending=False,
                            inplace=True)

    return model_table


def main():
    train_file = '/Users/dmitrenkoandrey/PycharmProjects/ml_kaggle_competition1/resources/melanoma/train.csv'
    test_file = '/Users/dmitrenkoandrey/PycharmProjects/ml_kaggle_competition1/resources/melanoma/test.csv'

    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    train.columns = [
        'img_name', 'id', 'sex', 'age', 'location', 'diagnosis',
        'benign_malignant', 'target'
    ]
    test.columns = ['img_name', 'id', 'sex', 'age', 'location']
    train.sample(5)
    test.sample(5)
    for df in [train, test]:
        df['location'].fillna('unknown', inplace=True)

    ids_train = train.location.values
    ids_test = test.location.values
    ids_train_set = set(ids_train)
    ids_test_set = set(ids_test)

    location_not_overlap = list(ids_train_set.symmetric_difference(ids_test_set))
    n_overlap = len(location_not_overlap)
    if n_overlap == 0:
        print(
            'There are no different body parts occuring between train and test set...'
        )
    else:
        print('There are some not overlapping values between train and test set!')
    train['sex'].fillna(train['sex'].mode()[0], inplace=True)

    train['age'].fillna(train['age'].median(), inplace=True)
    train40 = pd.read_csv('/Users/dmitrenkoandrey/PycharmProjects/ml_kaggle_competition1/resources/melanoma/add_features/train40Features.csv')
    test40 = pd.read_csv('/Users/dmitrenkoandrey/PycharmProjects/ml_kaggle_competition1/resources/melanoma/add_features/test40Features.csv')

    trainmet = pd.read_csv('/Users/dmitrenkoandrey/PycharmProjects/ml_kaggle_competition1/resources/melanoma/add_features/trainMetrics.csv')
    testmet = pd.read_csv('/Users/dmitrenkoandrey/PycharmProjects/ml_kaggle_competition1/resources/melanoma/add_features/testMetrics.csv')
    train40.drop(['sex', 'age_approx', 'anatom_site_general_challenge'],
                 axis=1,
                 inplace=True)

    test40.drop(['sex', 'age_approx', 'anatom_site_general_challenge'],
                axis=1,
                inplace=True)

    train = pd.concat([train, train40, trainmet], axis=1)
    test = pd.concat([test, test40, testmet], axis=1)

    train.head()
    # getting dummy variables for gender on train set

    sex_dummies = pd.get_dummies(train['sex'], prefix='sex')
    train = pd.concat([train, sex_dummies], axis=1)

    # getting dummy variables for gender on test set

    sex_dummies = pd.get_dummies(test['sex'], prefix='sex')
    test = pd.concat([test, sex_dummies], axis=1)

    # dropping not useful columns

    train.drop(['sex', 'img_name', 'id', 'diagnosis', 'benign_malignant'], axis=1, inplace=True)
    test.drop(['sex', 'img_name', 'id'], axis=1, inplace=True)
    # getting dummy variables for location on train set

    anatom_dummies = pd.get_dummies(train['location'], prefix='anatom')
    train = pd.concat([train, anatom_dummies], axis=1)

    # getting dummy variables for location on test set

    anatom_dummies = pd.get_dummies(test['location'], prefix='anatom')
    test = pd.concat([test, anatom_dummies], axis=1)

    # dropping not useful columns

    train.drop('location', axis=1, inplace=True)
    test.drop('location', axis=1, inplace=True)

    # dividing train set and labels for modelling

    X = train.drop('target', axis=1)
    y = train.target
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        stratify=y,
                                                        random_state=42)

    # 5 fold stratify for cv

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    xg = xgb.XGBClassifier(
        n_estimators=750,
        min_child_weight=0.81,
        learning_rate=0.025,
        max_depth=2,
        subsample=0.80,
        colsample_bytree=0.42,
        gamma=0.10,
        random_state=42,
        n_jobs=-1,
    )
    estimators = [xg]
    raw_models = model_check(X_train, y_train, estimators, cv)
    xg.fit(X_train, y_train)

    # predicting on holdout set
    validation = xg.predict_proba(X_test)[:, 1]

    # checking results on validation set
    roc_auc_score(y_test, validation)
    predictions = xg.predict_proba(test)[:, 1]
    # creating submission df

    meta_df = pd.DataFrame(columns=['image_name', 'target'])

    # assigning predictions on submission df
    sample = pd.read_csv('/Users/dmitrenkoandrey/PycharmProjects/ml_kaggle_competition1/resources/melanoma/add_features/sample_submission.csv')

    meta_df['image_name'] = sample['image_name']
    meta_df['target'] = predictions
    meta_df.to_csv('/Users/dmitrenkoandrey/PycharmProjects/ml_kaggle_competition1/result/submission_features_preds.csv', header=True, index=False)


if __name__ == "__main__":
    main()
