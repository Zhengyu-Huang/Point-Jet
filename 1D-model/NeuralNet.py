import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectNet_20(nn.Module):

    def __init__(self, N_in, N_out):
        super(DirectNet_20, self).__init__()
        # an affine operation: y = Wx + b
        
        self.fc1 = nn.Linear(N_in, 20)
        self.fc5 = nn.Linear(20, N_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


