import os
import sys
from pickle import dump

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as nnf
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from config import MEAN_MODEL_PARAMS, MODEL_PATH
from sampler import SoundDatasetSampler
from sound_dataset_random import sampler_label_callback, MeanDatasetFull


class Net(nn.Module):
    def __init__(self, n_features):
        super(Net, self).__init__()
        self.l1 = nn.Linear(n_features, 128)
        self.h11 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 9)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.h11(x))
        x = self.out(x)
        return x


if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

models_info = []


def train(data_folder):
    df = pd.read_csv(os.path.join(data_folder, 'train_ground_truth.csv'), dtype={0: str, 1: str},
                     names=['name', 'target'])

    # fit data
    scaler = fitX(df)
    # end fit data
    dump(scaler, open('/wdata/scaler.pkl', 'wb'))

    skf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (idxT, idxV) in enumerate(skf.split(df)):
        train = df.iloc[idxT]
        validation = df.iloc[idxV]
        params = MEAN_MODEL_PARAMS[fold]

        train_set = MeanDatasetFull(train)
        # train_set = MeanDatasetRandom(train, params)
        val_set = MeanDatasetFull(validation)
        # val_set = MeanDatasetRandom(validation, params)

        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=params['TRAIN_BATCH'], shuffle=False,
                                                   sampler=SoundDatasetSampler(train_set,
                                                                               callback_get_label=sampler_label_callback),
                                                   num_workers=4
                                                   )
        test_loader = torch.utils.data.DataLoader(val_set, batch_size=params['VALID_BATCH'], shuffle=True,
                                                  num_workers=4)
        net_model = Net(25)
        net_model.to(device)
        print(net_model)
        train_losses = []
        best_auc = 0.0
        optimizer = optim.Adam(net_model.parameters())
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        loss_f = nn.CrossEntropyLoss()
        for epoch in tqdm(range(1, params['EPOCHS'])):
            trace_y, trace_yhat = doTrain(epoch, loss_f, net_model, optimizer, train_loader, train_losses, scaler)
            auc = roc_auc_score(trace_y, trace_yhat,
                                multi_class="ovr", labels=[0, 1, 2, 3, 4, 5, 6, 7, 8])
            print('AUC : {:.6f}'.format(auc))
            scheduler.step()
            # VALIDATION
            valid_trace_y, valid_trace_yhat = doValidation(epoch, net_model, test_loader, scaler)
            accuracy = np.mean(valid_trace_yhat.argmax(axis=1) == valid_trace_y)
            auc = roc_auc_score(valid_trace_y, valid_trace_yhat, multi_class="ovr",
                                labels=[0, 1, 2, 3, 4, 5, 6, 7, 8])
            print(f'Epoch - {epoch} Valid-Accuracy : {accuracy}')
            print('AUC : {:.6f}'.format(auc))
            # END VALIDATION
            if auc > best_auc:
                print('!!!!!!!!!save best model auc is ' + str(auc))
                torch.save(net_model, MODEL_PATH + '_' + params['NAME'])
                best_auc = auc

        models_info.append({'fold': fold, 'name': params['NAME'], 'auc': best_auc})
    print(models_info)


def fitX(df):
    fit_set = MeanDatasetFull(df)
    fit_loader = torch.utils.data.DataLoader(fit_set,
                                             batch_size=32, shuffle=False,
                                             num_workers=4
                                             )
    scaler = StandardScaler()
    for batch_idx, (data, target) in enumerate(fit_loader):
        scaler.fit(np.array(data.numpy(), dtype=float))
    return scaler


def doValidation(epoch, net_model, test_loader, scaler):
    net_model.eval()
    valid_batch_losses = []
    valid_trace_y = []
    valid_trace_yhat = []
    valid_loss = 1.0
    for batch_idx, (batches, target) in enumerate(test_loader):
        batches = torch.tensor(scaler.transform(batches.numpy()))
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
    return valid_trace_y, valid_trace_yhat


def doTrain(epoch, loss_f, net_model, optimizer, train_loader, train_losses, scaler):
    net_model.train()
    trace_y = []
    trace_yhat = []
    batch_losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data = torch.tensor(scaler.transform(data.numpy()))
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
    return trace_y, trace_yhat


def main():
    train(sys.argv[1])


if __name__ == "__main__":
    main()
