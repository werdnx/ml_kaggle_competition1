import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torchvision.models import resnet34
from tqdm import tqdm

from utils import ESC50Data

df = pd.read_csv('/input/ESC-50-master/meta/esc50.csv')
msk = np.random.rand(len(df)) < 0.8
train_df = df[msk]
valid_df = df[~msk]

train_data = ESC50Data('/input/ESC-50-master/audio', train_df, 'filename', 'category')
valid_data = ESC50Data('/input/ESC-50-master/audio', valid_df, 'filename', 'category')
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=16, shuffle=True)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
resnet_model = resnet34(pretrained=True)
resnet_model.fc = nn.Linear(512, 50)
resnet_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
resnet_model = resnet_model.to(device)

learning_rate = 2e-4
optimizer = optim.Adam(resnet_model.parameters(), lr=learning_rate)
epochs = 50
loss_fn = nn.CrossEntropyLoss()
resnet_train_losses = []
resnet_valid_losses = []


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


def train(model, loss_fn, train_loader, valid_loader, epochs, optimizer, train_losses, valid_losses, change_lr=None):
    for epoch in tqdm(range(1, epochs + 1)):
        model.train()
        batch_losses = []
        if change_lr:
            optimizer = change_lr(optimizer, epoch)
        for i, data in enumerate(train_loader):
            x, y = data
            optimizer.zero_grad()
            x = x.to(device, dtype=torch.float32)
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


def main():
    train(resnet_model, loss_fn, train_loader, valid_loader, epochs, optimizer, resnet_train_losses,
          resnet_valid_losses,
          lr_decay)


if __name__ == "__main__":
    main()
