import csv
import os

import librosa
import numpy as np
import pandas as pd
import seaborn as sns
# import tensorflow as tf
# from tensorflow.python.keras.models import Sequential
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim
from dataset import FeatureDataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm
from xgb import train

from net import Net
from sampler import SoundDatasetSampler
from sampler import sampler_label_callback

# warnings.filterwarnings('ignore')
# AUTO = tf.data.experimental.AUTOTUNE
sns.set()
WAV_DIR = '/wdata/train/'
TEST_PATH = '/wdata/test/'
CV_TRAIN = 'dataset.csv'
CV_TEST = 'dataset_test.csv'
OUT_DIR = '/output/'
EPOCHS = 20
BATCH_SIZE = 32
labels = 9
MODEL_NAME = 'conv1d_fft_model_v1'
DF = '/data/train/train_ground_truth.csv'
DF_TEST = '/data/train/train_ground_truth.csv'
SAMPLING_RATE = 22050
GROUP_PATH = '/wdata/distribution-train-out.txt'
TEMP = '/wdata/temp/'
MODEL_PATH = '/wdata/model/trained_model'
CATEGORIES = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8}

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


def print_loss(resnet_train_losses):
    for loss in resnet_train_losses:
        print(f'\t {np.mean(loss)}')


def read_groups(df):
    f = open(GROUP_PATH, "r")
    lines = f.read().splitlines()
    result = []
    for line in lines:
        # r_l = []
        items = line.split(",")
        item = items[0]
        #     r_l.append(item)
        # result.append(r_l)
        row = df[df['name'] == item]
        # print('line is ')
        # print(line)
        # print('line end ')
        # print(row)
        row = row.iloc[0]
        result.append({"line": line, "target": row[1]})
    return pd.DataFrame(result)


def create_df(grouped_df):
    result = []
    for ind in range(len(grouped_df)):
        row = grouped_df.iloc[ind]
        names = row[0].split(",")
        for name in names:
            result.append({"name": name, "target": row[1]})
    return pd.DataFrame(result)


PREPARE_TRAIN = False
PREPARE_TEST = False
DO_TRAIN = False
DO_TEST = False

DO_XGB = True


def main():
    if PREPARE_TRAIN:
        prepare_data(DF, WAV_DIR, CV_TRAIN)

    data = pd.read_csv(TEMP + 'dataset.csv', header=None)
    data.head()  # Dropping unneccesary columns
    # data = data.drop(['filename'], axis=1)  # Encoding the Labels
    y_list = data.iloc[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_list)  # Scaling the Feature columns
    scaler = StandardScaler()
    X = scaler.fit_transform(
        np.array(data.iloc[:, :-1], dtype=float))  # Dividing data into training and Testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    if DO_TRAIN:
        train_set = FeatureDataset(X_train, y_train)
        valid_set = FeatureDataset(X_test, y_test)
        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=BATCH_SIZE, shuffle=False,
                                                   sampler=SoundDatasetSampler(train_set,
                                                                               callback_get_label=sampler_label_callback),
                                                   num_workers=1
                                                   )
        test_loader = torch.utils.data.DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=True,
                                                  num_workers=1)

        # print(X_train)

        net_model = Net(X_train.shape[1])
        net_model.to(device)
        print(net_model)
        train_losses = []
        valid_losses = []
        best_auc = 0.0
        optimizer = optim.Adam(net_model.parameters())
        # optimizer = torch.optim.SGD(net_model.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        loss_f = nn.CrossEntropyLoss()
        for epoch in tqdm(range(1, 101)):
            net_model.train()
            trace_y = []
            trace_yhat = []
            batch_losses = []

            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                data = data.to(device)
                target = target.to(device)
                data = data.requires_grad_()  # set requires_grad to True for training
                output = net_model(data)
                loss = loss_f(output, target)
                batch_losses.append(loss.item())
                loss.backward()
                optimizer.step()
                trace_y.append(target.cpu().detach().numpy())
                trace_yhat.append(nnf.softmax(output, dim=1).data.cpu().detach().numpy())
                if batch_idx % 10 == 0:  # print training stats
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss))

            train_losses.append(batch_losses)
            print(f'Epoch - {epoch} Train-Loss : {np.mean(train_losses[-1])}')
            trace_y = np.concatenate(trace_y)
            trace_yhat = np.concatenate(trace_yhat)
            auc = roc_auc_score(trace_y, trace_yhat,
                                multi_class="ovr", labels=[0, 1, 2, 3, 4, 5, 6, 7, 8])
            print('AUC : {:.6f}'.format(auc))

            scheduler.step()
            # VALIDATION
            net_model.eval()
            valid_batch_losses = []
            valid_trace_y = []
            valid_trace_yhat = []
            valid_loss = 1.0
            for batch_idx, (batches, target) in enumerate(test_loader):
                with torch.no_grad():
                    target = target.to(device)
                    batches = batches.float()
                    batches = batches.to(device)
                    output = net_model(batches)
                    valid_trace_y.append(target.cpu().detach().numpy())
                    valid_trace_yhat.append(nnf.softmax(output, dim=1).cpu().detach().numpy())
                    if batch_idx % 10 == 0:  # print training stats
                        print('Validation Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, batch_idx * len(batches), len(test_loader.dataset),
                                   100. * batch_idx / len(test_loader), valid_loss))

            valid_trace_y = np.concatenate(valid_trace_y)
            valid_trace_yhat = np.concatenate(valid_trace_yhat)
            accuracy = np.mean(valid_trace_yhat.argmax(axis=1) == valid_trace_y)
            auc = roc_auc_score(valid_trace_y, valid_trace_yhat, multi_class="ovr",
                                labels=[0, 1, 2, 3, 4, 5, 6, 7, 8])
            print(f'Epoch - {epoch} Valid-Accuracy : {accuracy}')
            print('AUC : {:.6f}'.format(auc))

            # END VALIDATION
            if auc > best_auc:
                print('!!!!!!!!!save best model auc is ' + str(auc))
                torch.save(net_model, MODEL_PATH + '_feature_model')
                best_auc = auc
    # print('train losses stat:')
    # print_loss(train_losses)
    #     TEST!!!!!!

    if PREPARE_TEST:
        file = open(TEMP + CV_TEST, 'w', newline='')
        file.close()
        files_to_predict = [(os.path.join(TEST_PATH, i), i) for i in os.listdir(TEST_PATH)]
        for file_name in tqdm(files_to_predict):
            to_append = create_features(file_name[0])
            to_append += f' {file_name[1].split(".")[0]}'
            file = open(TEMP + CV_TEST, 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())

    data = pd.read_csv(TEMP + CV_TEST, dtype=str, header=None)
    print('data len')
    print(str(len(data)))
    data.head()
    scaler = StandardScaler()
    y_list = data.iloc[:, -1]
    X = scaler.fit_transform(
        np.array(data.iloc[:, :-1], dtype=float))  # Dividing data into training and Testing set

    if DO_TEST:
        model = torch.load(MODEL_PATH + '_feature_model')
        model.to(device)
        model.eval()
        result = []
        print('len')
        print(str(len(X)))
        for ind in tqdm(range(len(X))):
            y = y_list[ind]
            XX = X[ind]
            XX = torch.tensor(XX).float()
            XX = XX[None, ...]
            XX = XX.to(device)
            output = model(XX)
            # print(output)
            probs = nnf.softmax(output, dim=1).cpu().detach().numpy()
            probs = probs[0]
            result.append(
                {"id": "{}".format(y), "A": probs[0], "B": probs[1],
                 "C": probs[2], "D": probs[3],
                 "E": probs[4],
                 "F": probs[5], "G": probs[6], "H": probs[7],
                 "I": probs[8]})
        df = pd.DataFrame(result)
        print(df.head())
        df.to_csv(TEMP + 'solution.csv', header=True, index=False, float_format='%.2f')

    if DO_XGB:
        train(X_train, y_train, X_test, y_test, X, y_list)


# END TEST


def prepare_data(path, wav_dir, cv_name):
    df = pd.read_csv(path, dtype={0: str, 1: str}, names=['name', 'target'])
    df = df.sample(frac=1).reset_index(drop=True)
    # groupsDf = read_groups(df)
    # train_df, valid_df = train_test_split(groupsDf, test_size=0.2, stratify=groupsDf['target'].to_numpy())
    # train_df = create_df(train_df)
    # valid_df = create_df(valid_df)
    file = open(TEMP + cv_name, 'w', newline='')
    file.close()
    for ind in tqdm(range(len(df))):
        row = df.iloc[ind]
        filename = wav_dir + row[0] + '.wav'
        to_append = create_features(filename)
        to_append += f' {CATEGORIES[row[1]]}'
        file = open(TEMP + cv_name, 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())


def create_features(filename):
    y, sr = librosa.load(filename, mono=True)
    # rmse = librosa.feature.rmse(y=y)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    to_append = f'{np.mean(chroma_stft)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
    for e in mfcc:
        to_append += f' {np.mean(e)}'
    return to_append


if __name__ == "__main__":
    main()
