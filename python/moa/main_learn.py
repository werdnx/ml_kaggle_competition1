import warnings
from collections import Counter

import category_encoders as ce
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from imblearn.over_sampling import SVMSMOTE
# from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from xgboost import XGBClassifier

# np.set_printoptions(threshold=sys.maxsize)

warnings.filterwarnings('ignore')

SEED = 42
NFOLDS = 5
DATA_DIR = '/input/'
np.random.seed(SEED)


def init_models(y_len):
    models = [None] * y_len
    for model_id in range(0, y_len):
        models[model_id] = create_model()
    return models

def create_def_model():
    m = Pipeline([('encode', ce.CountEncoder(cols=[0, 2])),
                  ('classify', XGBClassifier(tree_method='gpu_hist'))
                  ])
    params = {'classify__estimator__colsample_bytree': 0.6522,
              'classify__estimator__gamma': 3.6975,
              'classify__estimator__learning_rate': 0.0503,
              'classify__estimator__max_delta_step': 2.0706,
              'classify__estimator__max_depth': 10,
              'classify__estimator__min_child_weight': 31.5800,
              'classify__estimator__n_estimators': 166,
              'classify__estimator__subsample': 0.8639
              }
    m.set_params(**params)
    return m


def create_model():
    m = Pipeline([('encode', ce.CountEncoder(cols=[0, 2])),
                  ('sampling', SVMSMOTE()),
                  ('classify', XGBClassifier(tree_method='gpu_hist'))
                  ])
    params = {'classify__estimator__colsample_bytree': 0.6522,
              'classify__estimator__gamma': 3.6975,
              'classify__estimator__learning_rate': 0.0503,
              'classify__estimator__max_delta_step': 2.0706,
              'classify__estimator__max_depth': 10,
              'classify__estimator__min_child_weight': 31.5800,
              'classify__estimator__n_estimators': 166,
              'classify__estimator__subsample': 0.8639
              }
    m.set_params(**params)
    return m


def findEx(arr):
    i = 0
    for i in range(0, len(arr)):
        if arr[i] == 1:
            i += 1
    return i


def main():
    train = pd.read_csv(DATA_DIR + 'train_features.csv')
    targets = pd.read_csv(DATA_DIR + 'train_targets_scored.csv')
    train.head()

    test = pd.read_csv(DATA_DIR + 'test_features.csv')
    sub = pd.read_csv(DATA_DIR + 'sample_submission.csv')
    test.head()

    # drop where cp_type==ctl_vehicle (baseline)
    ctl_mask = train.cp_type == 'ctl_vehicle'
    train = train[~ctl_mask]
    targets = targets[~ctl_mask]
    # drop id col
    X = train.iloc[:, 1:].to_numpy()
    X_test = test.iloc[:, 1:].to_numpy()
    y = targets.iloc[:, 1:].to_numpy()

    models = init_models(len(y[0]))
    oof_preds = np.zeros(y.shape)
    test_preds = np.zeros((test.shape[0], y.shape[1]))
    kf = KFold(n_splits=NFOLDS)
    for fn, (trn_idx, val_idx) in enumerate(kf.split(X, y)):
        print('Starting fold: ', fn)
        for model_id in range(0, len(y[0])):
            print('Starting model: ', str(model_id))
            X_train, X_val = X[trn_idx], X[val_idx]
            y_train, y_val = y[trn_idx][:, model_id], y[val_idx][:, model_id]
            if findEx(y_train) < 6 or findEx(y_val) < 6:
                print('skip empty model_id ' + str(model_id))
            else:
                print('y_train' + str(Counter(y_train)))
                print('y_val' + str(Counter(y_val)))
                try:
                    model = create_model()
                    model.fit(X_train, y_train)
                except ValueError:
                    model = create_def_model()
                    model.fit(X_train, y_train)
                val_preds_cur = model.predict_proba(X_val)
                val_preds_cur = np.array(val_preds_cur)[:, 1].T
                oof_preds[val_idx][:, model_id] = val_preds_cur

                preds_cur = model.predict_proba(X_test)
                preds_cur = np.array(preds_cur)[:, 1].T
                test_preds[:, model_id] += preds_cur / NFOLDS

    print('OOF log loss: ', log_loss(np.ravel(y), np.ravel(oof_preds)))
    # set control test preds to 0
    control_mask = [test['cp_type'] == 'ctl_vehicle']
    test_preds[control_mask] = 0
    sub.iloc[:, 1:] = test_preds
    sub.to_csv('/output/submission_206_models_SMOTE.csv', index=False)
    # sub.to_csv('/output/submission_206_models_v2.csv', index=False)


if __name__ == "__main__":
    main()
