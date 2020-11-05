import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import KFold
from tqdm import tqdm

from config import FOLDS, TRAIN_PATH, EPOCHS, MODEL_PATH, BATCH_SIZE_TRAIN, BATCH_SIZE_TEST
from loss_function import LabelSmoothingCrossEntropy
from resnet import resnet_1d_34
from sampler import SoundDatasetSampler
from sound_dataset import SoundDataset, sampler_label_callback

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


def doTrain(model, epoch, train_loader, optimizer, resnet_train_losses):
    loss_f = LabelSmoothingCrossEntropy()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.half()
        data = data.to(device)
        target = target.half()
        target = target.to(device)
        data = data.requires_grad_()  # set requires_grad to True for training
        output = model(data)
        # output = output.permute(1, 0, 2)  # original output dimensions are batchSizex1x10
        loss = F.nll_loss(output, target)  # the loss functions expects a batchSizex10 input
        resnet_train_losses.append(loss.item())
        # loss = loss_f(output[0], target)
        # loss = log_loss(output[0], target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:  # print training stats
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss))
    print(f'Epoch - {epoch} Train-Loss : {np.mean(resnet_train_losses[-1])}')


def validation(model, test_loader, resnet_valid_losses, epoch):
    model.eval()
    correct = 0
    batch_losses = []
    trace_y = []
    trace_yhat = []
    for batch_idx, (data, target) in enumerate(test_loader):
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        trace_y.append(target.cpu().detach().numpy())
        trace_yhat.append(output.cpu().detach().numpy())
        # output = output.permute(1, 0, 2)
        loss = F.nll_loss(output, target)
        resnet_valid_losses.append(loss.item())
        batch_losses.append(loss.item())

    trace_y = np.concatenate(trace_y)
    trace_yhat = np.concatenate(trace_yhat)
    accuracy = np.mean(trace_yhat.argmax(axis=1) == trace_y)
    print(f'Epoch - {epoch} Valid-Loss : {np.mean(resnet_valid_losses[-1])} Valid-Accuracy : {accuracy}')
    # pred = output.max(2)[1]  # get the index of the max log-probability
    # correct += pred.eq(target).cpu().sum().item()
    # if batch_idx % 10 == 0:
    #     print(' Loss: {:.6f}\n'.format(loss))
    # print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))


def train(data_folder):
    df = pd.read_csv(os.path.join(data_folder, 'train_ground_truth.csv'), dtype={0: str, 1: str})

    skf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    for fold, (idxT, idxV) in enumerate(skf.split(np.arange(len(df)))):
        # msk = np.random.rand(len(df)) < 0.7
        train_df = df[df.index.isin(idxT)]
        valid_df = df[df.index.isin(idxV)]

        train_set = SoundDataset(TRAIN_PATH, train_df)
        validation_set = SoundDataset(TRAIN_PATH, valid_df)
        print('fold ' + str(fold))
        print("Train set size: " + str(len(train_set)))
        print("Test set size: " + str(len(validation_set)))

        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=BATCH_SIZE_TRAIN, shuffle=False,
                                                   sampler=SoundDatasetSampler(train_set,
                                                                               callback_get_label=sampler_label_callback),
                                                   num_workers=4
                                                   )
        test_loader = torch.utils.data.DataLoader(validation_set, batch_size=BATCH_SIZE_TEST, shuffle=False,
                                                  num_workers=4)
        # net_model = Net()
        net_model = resnet_1d_34(pretrained=False, num_classes=9)
        fc = net_model.fc
        net_model.fc = nn.Sequential(
            nn.Dropout(0.2),
            fc,
            nn.LogSoftmax(dim=-1)
        )
        net_model.half()  # convert to half precision
        for layer in net_model.modules():
            if isinstance(layer, nn.BatchNorm1d):
                layer.float()
        net_model.to(device)
        print(net_model)
        optimizer = optim.Adam(net_model.parameters(), lr=0.01, weight_decay=0.0001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        resnet_train_losses = []
        resnet_valid_losses = []
        for epoch in tqdm(range(1, EPOCHS + 1)):
            if epoch == 31:
                print("First round of training complete. Setting learn rate to 0.001.")
            doTrain(net_model, epoch, train_loader, optimizer, resnet_train_losses)
            scheduler.step()
            validation(net_model, test_loader, resnet_valid_losses, epoch)

        torch.save(net_model, MODEL_PATH + '_fold' + str(fold))


def main():
    train(sys.argv[1])


if __name__ == "__main__":
    main()
