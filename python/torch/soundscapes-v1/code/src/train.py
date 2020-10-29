import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet34
from tqdm import tqdm

from utils import ESC50Data

TRAIN_PATH = '/wdata/train'
MODEL_PATH = '/wdata/model/trained_model'
learning_rate = 2e-4
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


def setlr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return optimizer


def lr_decay(optimizer, epoch):
    if epoch % 10 == 0:
        new_lr = learning_rate / (10 ** (epoch // 10))
        optimizer = setlr(optimizer, new_lr)
        print(f'Changed learning rate to {new_lr}')
    return optimizer


def doTrain(model, loss_fn, train_loader, valid_loader, epochs, optimizer, train_losses, valid_losses, change_lr=None):
    for epoch in tqdm(range(1, epochs + 1)):
        model.train()
        batch_losses = []
        if change_lr:
            optimizer = change_lr(optimizer, epoch)
        for i, data in enumerate(train_loader):
            x, y = data
            optimizer.zero_grad()
            x = x.to(device, dtype=torch.float16)
            y = y.to(device, dtype=torch.long)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            batch_losses.append(loss.item())
            optimizer.step()
        train_losses.append(batch_losses)
        print(f'Epoch - {epoch} Train-Loss : {np.mean(train_losses[-1])}')
        model.eval()
        batch_losses = []
        trace_y = []
        trace_yhat = []
        for i, data in enumerate(valid_loader):
            x, y = data
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            trace_y.append(y.cpu().detach().numpy())
            trace_yhat.append(y_hat.cpu().detach().numpy())
            batch_losses.append(loss.item())
        valid_losses.append(batch_losses)
        trace_y = np.concatenate(trace_y)
        trace_yhat = np.concatenate(trace_yhat)
        accuracy = np.mean(trace_yhat.argmax(axis=1) == trace_y)
        print(f'Epoch - {epoch} Valid-Loss : {np.mean(valid_losses[-1])} Valid-Accuracy : {accuracy}')


def train(data_folder):
    df = pd.read_csv(os.path.join(data_folder, 'train_ground_truth.csv'), dtype={0: str, 1: str})
    msk = np.random.rand(len(df)) < 0.7
    train_df = df[msk]
    valid_df = df[~msk]
    train_data = ESC50Data(TRAIN_PATH, train_df, 0, 1)
    valid_data = ESC50Data(TRAIN_PATH, valid_df, 0, 1)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=32, shuffle=True)
    resnet_model = resnet34(pretrained=True)
    #9 - num classes
    resnet_model.fc = nn.Linear(512, 9)
    resnet_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    resnet_model = resnet_model.to(device)
    optimizer = optim.Adam(resnet_model.parameters(), lr=learning_rate)
    epochs = 30
    loss_fn = nn.CrossEntropyLoss()
    resnet_train_losses = []
    resnet_valid_losses = []
    doTrain(resnet_model, loss_fn, train_loader, valid_loader, epochs, optimizer, resnet_train_losses,
            resnet_valid_losses,
            lr_decay)
    torch.save(resnet_model, MODEL_PATH)


def main():
    train(sys.argv[1])


if __name__ == "__main__":
    main()
