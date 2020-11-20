import torch.nn as nn
import torch.nn.functional as F


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
