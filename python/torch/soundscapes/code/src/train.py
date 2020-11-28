import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from tqdm import tqdm

from audioutils import get_random_samples_from_file
from config import TRAIN_PATH, MODEL_PATH, MODEL_PARAMS, HALF, GROUP_PATH
from sampler import SoundDatasetSampler
from sound_dataset import SoundDatasetValidation, sampler_label_callback
from sound_dataset_random import SoundDatasetRandom

np.set_printoptions(threshold=sys.maxsize)
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


def wrap(data):
    if HALF:
        return data.half()
    else:
        return data


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


def doTrain(model, epoch, train_loader, optimizer, resnet_train_losses):
    # loss_f = LabelSmoothingCrossEntropy()
    model.train()
    trace_y = []
    trace_yhat = []
    batch_losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data = wrap(data)
        data = data.to(device)
        # target = target.half()
        target = target.to(device)
        # data = data.requires_grad_()  # set requires_grad to True for training
        output = model(data)
        # output = output.permute(1, 0, 2)  # original output dimensions are batchSizex1x10
        loss = F.nll_loss(output, target)  # the loss functions expects a batchSizex10 input
        batch_losses.append(loss.item())
        # loss = loss_f(output[0], target)
        # loss = log_loss(output[0], target)
        loss.backward()
        optimizer.step()
        # print('target')
        # print(target.cpu().detach().numpy())
        # print('preds')
        # print(torch.exp(output).data.cpu().detach().numpy())
        trace_y.append(target.cpu().detach().numpy())
        trace_yhat.append(torch.exp(output).data.cpu().detach().numpy())

        if batch_idx % 50 == 0:  # print training stats
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss))

    resnet_train_losses.append(batch_losses)
    print(f'Epoch - {epoch} Train-Loss : {np.mean(resnet_train_losses[-1])}')
    trace_y = np.concatenate(trace_y)
    trace_yhat = np.concatenate(trace_yhat)
    auc = roc_auc_score(trace_y, trace_yhat,
                        multi_class="ovr", labels=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    print('AUC : {:.6f}'.format(auc))


def print_loss(resnet_train_losses):
    for loss in resnet_train_losses:
        print(f'\t {np.mean(loss)}')


def validation_group(model, test_loader, resnet_valid_losses, epoch, model_param):
    model.eval()
    batch_losses = []
    trace_y = []
    trace_yhat = []
    for batch_idx, (crops_batches, target) in enumerate(test_loader):
        with torch.no_grad():
            # target = target.to(device)

            # probs = torch.zeros(len(crops_batches), 9)
            # probs = probs.to(device)
            probs = None
            for crop_batch_idx, file_path in enumerate(crops_batches):
                crops = get_random_samples_from_file(file_path, model_param['SECONDS'])
                data = None
                for crop in crops:
                    if data is None:
                        data = crop[np.newaxis, ...][None, ...]
                    else:
                        data = torch.cat((data, crop[np.newaxis, ...][None, ...]), dim=0)
                    # print(data)
                # crops = crops[None, ...]
                crops = data
                crops = wrap(crops)
                crops = crops.to(device)
                output = model(crops)
                prob = torch.mean(torch.exp(output).cpu().detach(), dim=0)
                if probs is None:
                    probs = prob[None, ...]
                else:
                    probs = torch.cat((probs, prob[None, ...]), dim=0)
                crops.detach()

                # for crop in crops:
                #     crop = crop[np.newaxis, ...]
                #     crop = crop[None, ...]
                #     crop = wrap(crop)
                #     crop = crop.to(device)
                #     output = model(crop)
                #     probs[crop_batch_idx] = torch.add(probs[crop_batch_idx], output.cpu().detach())
                #     crop.detach()
                # probs[crop_batch_idx] = probs[crop_batch_idx] / float(len(crops))

            # trace_y.append(target.cpu().detach().numpy())
            trace_y.append(target.numpy())
            trace_yhat.append(probs.numpy())
            # trace_yhat.append(probs.cpu().detach().numpy())
            # print('probs:')
            # print(log_probs)
            # print('target:')
            # print(target)
            loss = 1.0
            # loss = F.nll_loss(log_probs, target)
            # print('prooobbs')
            # print(torch.exp(log_probs).numpy())

            # batch_losses.append(loss.item)
            # batch_losses.append(loss.item())
            # target.detach()
            # probs.detach()
            if batch_idx % 10 == 0:  # print training stats
                print('Validation Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(crops_batches), len(test_loader.dataset),
                           100. * batch_idx / len(test_loader), loss))

            # crops_batches = crops_batches.half()
            # crops_batches = crops_batches.to(device)
            # output = model(crops_batches)
            # trace_y.append(target.cpu().detach().numpy())
            # trace_yhat.append(output.cpu().detach().numpy())
            # loss = F.nll_loss(output, target)
            # batch_losses.append(loss.item())
            # if batch_idx % 50 == 0:  # print training stats
            #     print('Validation Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(crops_batches), len(test_loader.dataset),
            #                100. * batch_idx / len(test_loader), loss))

    resnet_valid_losses.append(batch_losses)
    trace_y = np.concatenate(trace_y)
    trace_yhat = np.concatenate(trace_yhat)
    accuracy = np.mean(trace_yhat.argmax(axis=1) == trace_y)
    auc = roc_auc_score(trace_y, trace_yhat, multi_class="ovr",
                        labels=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    print(f'Epoch - {epoch} Valid-Loss : {np.mean(resnet_valid_losses[-1])} Valid-Accuracy : {accuracy}')
    print('AUC : {:.6f}'.format(auc))
    return auc
    # return np.mean(resnet_valid_losses[-1])
    # pred = output.max(2)[1]  # get the index of the max log-probability
    # correct += pred.eq(target).cpu().sum().item()
    # if batch_idx % 10 == 0:
    #     print(' Loss: {:.6f}\n'.format(loss))
    # print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))


def validation(model, test_loader, resnet_valid_losses, epoch, model_param):
    model.eval()
    batch_losses = []
    trace_y = []
    trace_yhat = []
    for batch_idx, (crops_batches, target) in enumerate(test_loader):
        with torch.no_grad():
            target = target.to(device)
            crops_batches = wrap(crops_batches)
            crops_batches = crops_batches.to(device)
            output = model(crops_batches)
            trace_y.append(target.cpu().detach().numpy())
            trace_yhat.append(output.cpu().detach().numpy())
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


def train(data_folder):
    df = pd.read_csv(os.path.join(data_folder, 'train_ground_truth.csv'), dtype={0: str, 1: str},
                     names=['name', 'target'])
    # files = [(os.path.join(TRAIN_PATH, i), i) for i in os.listdir(TRAIN_PATH)]
    # to_train = []
    # for f in files:
    #     name = f[1].split(".")[0]
    #     row = df[df['name'] == name]
    #     row = row.iloc[0]
    #     print(row)
    # to_train.append({'name': row[0], 'target': row[1]})
    # df = pd.DataFrame(to_train)
    print(df.head())
    print('len of train df ' + str(len(df)))
    # {line, target}
    # groupsDf = read_groups(df)
    # print(groupsDf.head())
    models_info = []
    df = df.sample(frac=1).reset_index(drop=True)
    skf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (idxT, idxV) in enumerate(skf.split(df)):
        train_df = df.iloc[idxT]
        valid_df = df.iloc[idxV]
        model_param = MODEL_PARAMS[fold]

        # for model_param in MODEL_PARAMS:

        # df = df[:100]
        # train_df, valid_df = train_test_split(df, test_size=0.2, stratify=df['target'].to_numpy())
        # train_df, valid_df = train_test_split(df, test_size=0.2, stratify=df['target'].to_numpy())
        print(valid_df.head())
        print(train_df.head())
        train_df = create_df(train_df)
        valid_df = create_df(valid_df)

        # msk = np.random.rand(len(df)) < 0.7
        # train_df = df[msk]
        # valid_df = df[~msk]

        # train_df = df[df.index.isin(idxT)]
        # valid_df = df[df.index.isin(idxV)]

        # TODO remove fo debug purpose
        # train_df = train_df[:100]
        # valid_df = valid_df[:100]

        train_set = SoundDatasetRandom(TRAIN_PATH, train_df, model_param)
        validation_set = SoundDatasetValidation(TRAIN_PATH, valid_df, model_param)
        print('model: ' + str(model_param['NAME']))
        print("Train set size: " + str(len(train_set)))
        print("Test set size: " + str(len(validation_set)))

        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=model_param['TRAIN_BATCH'], shuffle=False,
                                                   sampler=SoundDatasetSampler(train_set,
                                                                               callback_get_label=sampler_label_callback),
                                                   num_workers=4
                                                   )
        test_loader = torch.utils.data.DataLoader(validation_set, batch_size=model_param['VALID_BATCH'], shuffle=True,
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
        net_model = wrap(net_model)  # convert to half precision
        for layer in net_model.modules():
            if isinstance(layer, nn.BatchNorm1d):
                layer.float()
        net_model.to(device)
        print(net_model)
        # optimizer = optim.SGD(net_model.parameters(), lr=1e-3, momentum=0.9)
        optimizer = optim.Adam(net_model.parameters())
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        resnet_train_losses = []
        resnet_valid_losses = []
        best_auc = 0.0
        for epoch in tqdm(range(1, model_param['EPOCHS'] + 1)):
            doTrain(net_model, epoch, train_loader, optimizer, resnet_train_losses)
            scheduler.step()
            auc = validation_group(net_model, test_loader, resnet_valid_losses, epoch, model_param)
            if auc > best_auc:
                print('!!!!!!!!!save best model ' + model_param['NAME'])
                torch.save(net_model, MODEL_PATH + '_' + model_param['NAME'])
                best_auc = auc
        print('train losses stat:')
        print_loss(resnet_train_losses)
        print('validation losses stat:')
        print_loss(resnet_valid_losses)
        models_info.append({'fold': fold, 'name': model_param['NAME'], 'auc': best_auc})
        # torch.save(net_model, MODEL_PATH + '_fold' + str(fold))
    print(models_info)


def create_df(grouped_df):
    result = []
    for ind in range(len(grouped_df)):
        row = grouped_df.iloc[ind]
        names = row[0].split(",")
        for name in names:
            result.append({"name": name, "target": row[1]})
    return pd.DataFrame(result)


def main():
    train(sys.argv[1])


if __name__ == "__main__":
    main()
