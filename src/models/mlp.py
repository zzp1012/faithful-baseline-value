import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, in_dim, hidd_dim, out_dim):
        super(MLP, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.fc1 = nn.Linear(in_dim,hidd_dim)
        self.fc2 = nn.Linear(hidd_dim,hidd_dim)
        self.fc3 = nn.Linear(hidd_dim, out_dim)

    def forward(self, x, return_feature = False):
        x = x.reshape(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        if return_feature:
            return x
        x = F.relu(x)
        x = self.fc3(x)
        return x