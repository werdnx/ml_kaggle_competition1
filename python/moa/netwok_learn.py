import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ReduceLROnPlateau

# EPOCHS = 100
# BATCH_SIZE = 2048
BATCH_SIZE = 128
EPOCHS = 35


def seed_everything(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


seed_everything(42)


def model():
    METRICS = [
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'),
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
    ]
    model = tf.keras.Sequential([
        tf.keras.layers.Input(877),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(4096, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(4096, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(206, activation="sigmoid")
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=2.75e-5), loss='binary_crossentropy',
                  metrics=["accuracy", "AUC"])
    return model


def main():
    train_features = pd.read_csv('/input/train_features.csv')
    train_targets = pd.read_csv('/input/train_targets_scored.csv')
    COLS = ['cp_type', 'cp_dose']
    FE = []
    for col in COLS:
        for mod in train_features[col].unique():
            FE.append(mod)
            train_features[mod] = (train_features[col] == mod).astype(int)
    del train_features['sig_id']
    del train_features['cp_type']
    del train_features['cp_dose']
    FE += list(train_features.columns)
    del train_targets['sig_id']

    NFOLD = 5
    kf = KFold(n_splits=NFOLD)

    test_features = pd.read_csv('../input/test_features.csv')
    for col in COLS:
        for mod in test_features[col].unique():
            test_features[mod] = (test_features[col] == mod).astype(int)
    sig_id = pd.DataFrame()
    sig_id = test_features.pop('sig_id')
    del test_features['cp_type']
    del test_features['cp_dose']

    pe = np.zeros((test_features.shape[0], 206))

    train_features = train_features.values
    train_targets = train_targets.values
    pred = np.zeros((train_features.shape[0], 206))

    cnt = 0
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        verbose=1,
        patience=10,
        mode='max',
        restore_best_weights=True)

    for tr_idx, val_idx in kf.split(train_features):
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, epsilon=1e-4,
                                           mode='min')
        cnt += 1
        print("FOLD " + str(cnt))
        net = model()
        net.fit(train_features[tr_idx], train_targets[tr_idx], batch_size=BATCH_SIZE, epochs=EPOCHS,
                validation_data=(train_features[val_idx], train_targets[val_idx]), verbose=1,
                callbacks=[reduce_lr_loss, early_stopping])
        print("train", net.evaluate(train_features[tr_idx], train_targets[tr_idx], verbose=1, batch_size=BATCH_SIZE))
        print("val", net.evaluate(train_features[val_idx], train_targets[val_idx], verbose=1, batch_size=BATCH_SIZE))
        print("predict val...")
        pred[val_idx] = net.predict(train_features[val_idx], batch_size=BATCH_SIZE, verbose=0)
        print("predict test...")
        pe += net.predict(test_features, batch_size=BATCH_SIZE, verbose=0) / NFOLD

    columns = pd.read_csv('../input/train_targets_scored.csv')
    del columns['sig_id']
    sub = pd.DataFrame(data=pe, columns=columns.columns)
    sample = pd.read_csv('../input/sample_submission.csv')
    sub.insert(0, column='sig_id', value=sample['sig_id'])
    sub.to_csv('/output/submission_NN_5_fold_v2.csv', index=False)


if __name__ == "__main__":
    main()
