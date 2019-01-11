import torch

import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# def _initialize_weights(modules):
#     for module in modules:
#         if isinstance(module, nn.Linear):
#             module.weight = nn.init.xavier_uniform_(module.weight)
#
#
# class LinearNet(nn.Module):
#     def __init__(self, input_size, hidden_layers, output_size):
#         super(LinearNet, self).__init__()
#
#         seq = self._create_layers(input_size, hidden_layers, output_size)
#         self.model = nn.Sequential(*seq)
#         _initialize_weights(self.modules())
#
#     def _create_layers(self, input_size, hidden_layers, output_size):
#         seq = [nn.Linear(input_size, hidden_layers[0])]
#         for i in range(1, len(hidden_layers)):
#             seq = seq + [nn.ReLU()]
#             seq = seq + [nn.Linear(hidden_layers[i - 1], hidden_layers[i])]
#
#         seq = seq + [nn.ReLU()]
#         seq = seq + [nn.Linear(hidden_layers[-1], output_size)]
#         return seq
#
#     def forward(self, state):
#         return self.model.forward(state)
#
#
# class Actor(LinearNet):
#     def __init__(self, input_size, hidden_layers, output_size):
#         super(Actor, self).__init__(input_size, hidden_layers, output_size)
#
#     def _create_layers(self, input_size, hidden_layers, output_size):
#         seq = super(Actor, self)._create_layers(input_size, hidden_layers, output_size)
#         seq = seq + [nn.Tanh()]
#         return seq
#
#
# class Critic(nn.Module):
#     def __init__(self, state_size, action_size, hidden_layers):
#         super(Critic, self).__init__()
#         self.state_size = state_size
#         self.action_size = action_size
#
#         self._create_layers(state_size, action_size, hidden_layers, 1)
#         _initialize_weights([self.fc1, self.fc2, self.fc3])
#
#     def _create_layers(self, state_size, action_size, hidden_layers, output_size):
#         self.fc1 = nn.Linear(in_features=state_size, out_features=hidden_layers[0])
#         self.fc2 = nn.Linear(in_features=hidden_layers[0] + action_size, out_features=hidden_layers[1])
#         self.fc3 = nn.Linear(in_features=hidden_layers[1], out_features=output_size)
#
#     def forward(self, state, action):
#         x = F.relu(self.fc1(state))
#         x = F.relu(self.fc2(torch.cat([x, action], dim=1)))
#         x = F.relu(self.fc3(x))
#         return x


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, fc1_units=256, fc2_units=128):
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
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, fc1_units=256, fc2_units=128):
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
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.leaky_relu(self.fc1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.leaky_relu(self.fc2(x))
        return self.fc3(x)
