
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

import itertools


class ValueNetwork(nn.Module):

    def __init__(self, inp):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(inp, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        state_dict = self.fc3.state_dict()
        state_dict["bias"] = (state_dict["bias"]*0).detach()
        state_dict["weight"] = (state_dict["weight"]*0).detach()
        self.fc3.load_state_dict(state_dict)
        # print([x for x in self.fc3.load_state_dict()])
        # print(state_dict)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.squeeze(x, 1)
        return x

class ActionNetwork(nn.Module):

    def __init__(self, inp, out):
        super(ActionNetwork, self).__init__()
        self.fc1 = nn.Linear(inp, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, out)
        state_dict = self.fc3.state_dict()
        state_dict["bias"] = (state_dict["bias"]*0).detach()
        state_dict["weight"] = (state_dict["weight"]*0).detach()
        self.fc3.load_state_dict(state_dict)
        # print([x for x in self.fc3.load_state_dict()])
        # print(state_dict)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.softmax(x, dim=1)
        return x
