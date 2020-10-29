import os
import sys

from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torchsampler import ImbalancedDatasetSampler
from net import Net
from sound_dataset import SoundDataset

TRAIN_PATH = '/wdata/train'
MODEL_PATH = '/wdata/model/trained_model'
FOLDS = 5
EPOCHS = 40

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


def doTrain(model, epoch, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        data = data.requires_grad_()  # set requires_grad to True for training
        output = model(data)
        output = output.permute(1, 0, 2)  # original output dimensions are batchSizex1x10
        loss = F.nll_loss(output[0], target)  # the loss functions expects a batchSizex10 input
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:  # print training stats
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss))


def validation(model, test_loader):
    model.eval()
    correct = 0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        output = output.permute(1, 0, 2)
        pred = output.max(2)[1]  # get the index of the max log-probability
        correct += pred.eq(target).cpu().sum().item()
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def train(data_folder):
    df = pd.read_csv(os.path.join(data_folder, 'train_ground_truth.csv'), dtype={0: str, 1: str})

    skf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    for fold, (idxT, idxV) in enumerate(skf.split(np.arange(len(df)))):
        # msk = np.random.rand(len(df)) < 0.7
        train_df = df[idxT]
        valid_df = df[idxV]

        train_set = SoundDataset(TRAIN_PATH, train_df)
        validation_set = SoundDataset(TRAIN_PATH, valid_df)
        print("Train set size: " + str(len(train_set)))
        print("Test set size: " + str(len(validation_set)))

        kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}  # needed for using datasets on gpu
        train_loader = torch.utils.data.DataLoader(train_set, sampler=ImbalancedDatasetSampler(train_set),
                                                   batch_size=128, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(validation_set, batch_size=128, shuffle=True, **kwargs)
        net_model = Net()
        net_model.to(device)
        print(net_model)
        optimizer = optim.Adam(net_model.parameters(), lr=0.01, weight_decay=0.0001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        for epoch in tqdm(range(1, EPOCHS + 1)):
            if epoch == 31:
                print("First round of training complete. Setting learn rate to 0.001.")
            scheduler.step()
            doTrain(net_model, epoch, train_loader, optimizer)
            validation(net_model, epoch, test_loader)

        torch.save(net_model, MODEL_PATH + '_fold' + str(fold))


def main():
    train(sys.argv[1])


if __name__ == "__main__":
    main()
