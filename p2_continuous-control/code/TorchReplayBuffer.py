import numpy as np
import torch

from ReplayBuffer import ReplayBuffer


class TorchReplayBuffer(ReplayBuffer):

    def __init__(self, state_size, action_size, buffer_capacity, device):
        ReplayBuffer.__init__(self, state_size, action_size, buffer_capacity)
        self.device = device

    def sample(self, n, exact_batch_size=False):
        states, actions, rewards, next_states, dones = ReplayBuffer.sample(self, n, exact_batch_size)

        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones.astype(np.uint8)).float().to(self.device)

        return states, actions, rewards, next_states, dones
