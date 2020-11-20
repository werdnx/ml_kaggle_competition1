import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from tqdm import tqdm

TEMP = '/wdata/temp/'


def train(X_train, y_train, X_val, y_val, X_test, test_file_names):
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
    validation = xg.predict_proba(X_val)[:, 1]

    # checking results on validation set
    # roc_auc_score(y_val, validation, multi_class="ovr", labels=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    predictions = xg.predict_proba(X_test)
    print(predictions)
    result = []
    for ind in tqdm(range(len(X_test))):
        probs = predictions[ind]
        result.append(
            {"id": "{}".format(test_file_names[ind]), "A": probs[0], "B": probs[1],
             "C": probs[2], "D": probs[3],
             "E": probs[4],
             "F": probs[5], "G": probs[6], "H": probs[7],
             "I": probs[8]})

    df = pd.DataFrame(result)
    print(df.head())
    df.to_csv(TEMP + 'solution.csv', header=True, index=False, float_format='%.2f')


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
                                    scoring='accuracy',
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
