import copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


def xavier_init(module):
    nn.init.xavier_uniform_(module.weight, gain=1)
    module.bias.data.fill_(0.1)


class ForwardNet(nn.Module):
    def __init__(self):
        super(ForwardNet, self).__init__()

    def forward(self, state):
        return state


class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        # self.bn1 = ForwardNet()
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        # self.bn2 = ForwardNet()
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

        self.drop_layer = nn.Dropout(p=0.2)
        self.raw_output = None

    def reset_parameters(self):
        xavier_init(self.fc1)
        xavier_init(self.fc2)
        xavier_init(self.fc3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x1 = F.leaky_relu(self.drop_layer(self.bn1(self.fc1(state))))
        x2 = F.leaky_relu(self.drop_layer(self.bn2(self.fc2(x1))))
        x3 = self.fc3(x2)
        self.raw_output = copy.copy(x3[0])
        # print("Raw output capture", self.raw_output)
        output = torch.tanh(x3)
        # output = self.raw_output

        # print("Input", state[0])
        # print("Layers[0].fc1", self.fc1(state2)[0])
        # print("Layers[0].bn1", self.bn1(self.fc1(state2))[0])
        # print("Layers[0].drop1", self.drop_layer(self.bn1(self.fc1(state2))[0]))
        # print("Layers[0].relu", x1[0])
        # print("Layers[1].fc2", self.fc2(x1)[0])
        # print("Layers[1].bn2", self.bn2(self.fc2(x1))[0])
        # print("Layers[1].drop2", self.drop_layer(self.bn2(self.fc2(x1)))[0])
        # print("Layers[1].relu", x2[0])
        # print("Layers[2].fc3", self.raw_output)
        # print("Layers[2].tanh", output[0])
        # print()
        return output

        # x = F.relu(self.drop_layer(self.fc1(state)))
        # x = F.relu(self.drop_layer(self.fc2(x)))
        # return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        # self.bn1 = ForwardNet()
        self.fc2 = nn.Linear(fc1_units + action_size, fc2_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        # self.bn2 = ForwardNet()
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

        self.drop_layer = nn.Dropout(p=0.2)

    def reset_parameters(self):
        xavier_init(self.fc1)
        xavier_init(self.fc2)
        xavier_init(self.fc3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.leaky_relu(self.drop_layer(self.bn1(self.fc1(state))))
        x = torch.cat((xs, action), dim=1)
        x = F.leaky_relu(self.drop_layer(self.bn2(self.fc2(x))))
        return self.fc3(x)

        # xs = F.relu(self.drop_layer(self.fc1(state)))
        # x = torch.cat((xs, action), dim=1)
        # x = F.relu(self.drop_layer(self.fc2(x)))
        # return self.fc3(x)
