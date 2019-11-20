import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.lin1 = nn.Linear(4, 100)
        self.lin2 = nn.Linear(100, 100)
        self.lin3 = nn.Linear(100, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        X = F.relu(self.lin1(X))
        X = F.relu( self.lin2(X))
        X = self.lin3(X)
        X = self.softmax(X)

        return X
