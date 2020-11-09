import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from efficientnet_pytorch import EfficientNet
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import TRAIN_PATH, MODEL_PATH, MODEL_PARAMS
from sampler import SoundDatasetSampler
from sound_dataset import sampler_label_callback, SoundDataset
from sound_dataset_random import SoundDatasetRandom

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


def doTrain(model, epoch, train_loader, optimizer, resnet_train_losses):
    # loss_f = LabelSmoothingCrossEntropy()
    model.train()
    batch_losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.half()
        data = data.to(device)
        # target = target.half()
        target = target.to(device)
        data = data.requires_grad_()  # set requires_grad to True for training
        output = model(data)
        # output = output.permute(1, 0, 2)  # original output dimensions are batchSizex1x10
        loss = F.nll_loss(output, target)  # the loss functions expects a batchSizex10 input
        batch_losses.append(loss.item())
        # loss = loss_f(output[0], target)
        # loss = log_loss(output[0], target)
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:  # print training stats
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss))
    resnet_train_losses.append(batch_losses)
    print(f'Epoch - {epoch} Train-Loss : {np.mean(resnet_train_losses[-1])}')


def validation(model, test_loader, resnet_valid_losses, epoch):
    model.eval()
    # correct = 0
    batch_losses = []
    trace_y = []
    trace_yhat = []
    for batch_idx, (crops_batches, target) in enumerate(test_loader):
        target = target.to(device)

        # for crops in crops_batches:
        #     crops = crops.half()
        #     crops = crops.to(device)
        #     output = model(crops)
        #     output = output.sum(0) / float(len(crops))

        crops_batches = crops_batches.half()
        crops_batches = crops_batches.to(device)
        output = model(crops_batches)
        trace_y.append(target.cpu().detach().numpy())
        trace_yhat.append(output.cpu().detach().numpy())
        # output = output.permute(1, 0, 2)
        loss = F.nll_loss(output, target)
        batch_losses.append(loss.item())
        if batch_idx % 50 == 0:  # print training stats
            print('Validation Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(crops_batches), len(test_loader.dataset),
                       100. * batch_idx / len(test_loader), loss))

    resnet_valid_losses.append(batch_losses)
    trace_y = np.concatenate(trace_y)
    trace_yhat = np.concatenate(trace_yhat)
    accuracy = np.mean(trace_yhat.argmax(axis=1) == trace_y)
    print(f'Epoch - {epoch} Valid-Loss : {np.mean(resnet_valid_losses[-1])} Valid-Accuracy : {accuracy}')

    return np.mean(resnet_valid_losses[-1])
    # pred = output.max(2)[1]  # get the index of the max log-probability
    # correct += pred.eq(target).cpu().sum().item()
    # if batch_idx % 10 == 0:
    #     print(' Loss: {:.6f}\n'.format(loss))
    # print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))


def print_loss(resnet_train_losses):
    for loss in resnet_train_losses:
        print(f'\t {np.mean(loss)}')


def train(data_folder):
    df = pd.read_csv(os.path.join(data_folder, 'train_ground_truth.csv'), dtype={0: str, 1: str})

    # skf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    # for fold, (idxT, idxV) in enumerate(skf.split(np.arange(len(df)))):
    for model_param in MODEL_PARAMS:
        df = df.sample(frac=1).reset_index(drop=True)
        train_df, valid_df = train_test_split(df, test_size=0.2)
        # msk = np.random.rand(len(df)) < 0.7
        # train_df = df[msk]
        # valid_df = df[~msk]

        # train_df = df[df.index.isin(idxT)]
        # valid_df = df[df.index.isin(idxV)]

        # TODO remove fo debug purpose
        # train_df = train_df[:100]
        # valid_df = valid_df[:100]

        train_set = SoundDatasetRandom(TRAIN_PATH, train_df, model_param)
        validation_set = SoundDataset(TRAIN_PATH, valid_df, model_param)
        print('model: ' + str(model_param['NAME']))
        print("Train set size: " + str(len(train_set)))
        print("Test set size: " + str(len(validation_set)))

        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=model_param['TRAIN_BATCH'], shuffle=False,
                                                   sampler=SoundDatasetSampler(train_set,
                                                                               callback_get_label=sampler_label_callback),
                                                   num_workers=4
                                                   )
        test_loader = torch.utils.data.DataLoader(validation_set, batch_size=model_param['VALID_BATCH'], shuffle=False,
                                                  num_workers=4)
        # model_ft = models.resnet152(pretrained=True)
        # num_ftrs = model_ft.fc.in_features
        # model_ft.fc = nn.Linear(num_ftrs, 120)
        net_model = EfficientNet.from_pretrained(model_param['TYPE'], in_channels=1)
        # net_model = resnet34(pretrained=True)
        # net_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = net_model._fc.in_features
        net_model._fc = nn.Sequential(
            # nn.Dropout(DROPOUT),
            nn.Linear(num_ftrs, 9),
            nn.LogSoftmax(dim=-1)
        )
        net_model.half()  # convert to half precision
        for layer in net_model.modules():
            if isinstance(layer, nn.BatchNorm1d):
                layer.float()
        net_model.to(device)
        print(net_model)
        optimizer = optim.SGD(net_model.parameters(), lr=1e-3, momentum=0.9)
        # optimizer = optim.Adam(net_model.parameters(), eps=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        resnet_train_losses = []
        resnet_valid_losses = []
        best_loss = 100.0
        for epoch in tqdm(range(1, model_param['EPOCHS'] + 1)):
            doTrain(net_model, epoch, train_loader, optimizer, resnet_train_losses)
            scheduler.step()
            loss = validation(net_model, test_loader, resnet_valid_losses, epoch)
            if loss < best_loss:
                print('!!!!!!!!!save best model ' + model_param['NAME'])
                torch.save(net_model, MODEL_PATH + '_' + model_param['NAME'])
                best_loss = loss
        print('train losses stat:')
        print_loss(resnet_train_losses)
        print('validation losses stat:')
        print_loss(resnet_valid_losses)
        # torch.save(net_model, MODEL_PATH + '_fold' + str(fold))


def main():
    train(sys.argv[1])


if __name__ == "__main__":
    main()
