import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F


class Net(Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(8, 500)  # 8 is the number of entries in each row
        self.fc2 = nn.Linear(500, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x