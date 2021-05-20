import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, action_size):
        super(Net, self).__init__()
        self.fc   = nn.Linear(obs_size, hidden_size)
        self.relu = nn.ReLU()
        self.out  = nn.Linear(hidden_size, action_size) 
    def forward(self, x):
        x   = self.fc(x)
        x   = self.relu(x)
        # out = F.softmax(self.out(x), 1)
        out = self.out(x)
        return out