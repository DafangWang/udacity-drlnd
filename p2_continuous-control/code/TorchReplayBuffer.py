import numpy as np
import torch


class TorchReplayBuffer:

    def __init__(self, delegate, device):
        self.delegate = delegate
        self.device = device

    def sample(self, samples_to_draw):
        states, actions, rewards, next_states, dones = self.delegate.sample(samples_to_draw)

        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones.astype(np.uint8)).float().to(self.device)

        return states, actions, rewards, next_states, dones

    def size(self):
        return self.delegate.size()

    def add(self, state, action, reward, next_state, done):
        return self.delegate.add(state, action, reward, next_state, done)

