import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectNet_20(nn.Module):

    def __init__(self, N_in, N_out):
        super(DirectNet_20, self).__init__()
        # an affine operation: y = Wx + b
        
        self.fc1 = nn.Linear(N_in, 20)
        self.fc2 = nn.Linear(20, N_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


