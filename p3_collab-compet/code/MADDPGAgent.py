import random

import numpy as np
import torch

from DDPGAgent import DDPGAgent, ReplayBuffer
from DecayingValue import ExponentialDecay, SpacedRepetitionDecay
from HyperParam import HyperParam

BUFFER_SIZE = int(1e6)
BATCH_SIZE = 128

EXPLORE_EXPLOIT_DECAY = 0.999

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MADDPGAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, num_agents, random_seed, hyper_param=None):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.num_agents = num_agents

        if hyper_param is None:
            hyper_param = HyperParam()
            hyper_param.epsilon = True
            hyper_param.epsilon_decay = EXPLORE_EXPLOIT_DECAY
            hyper_param.epsilon_spaced_init = 100
            hyper_param.epsilon_spaced_decay = 1.5

        self.hyper_param = hyper_param

        self.device = device
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

        self.agents = [DDPGAgent(state_size, action_size, random_seed, self.memory) for _ in range(num_agents)]

        self.epsilon = SpacedRepetitionDecay(ExponentialDecay(1.0, 0.0, hyper_param.epsilon_decay),
                                             hyper_param.epsilon_spaced_init, hyper_param.epsilon_spaced_decay)

        self.train_mode = True

    def step(self, time_step, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""

        for i in range(self.num_agents):
            self.memory.add(state[i], action[i], reward[i], next_state[i], done[i])

        for agent in self.agents:
            agent.step(time_step)

    def train(self, mode=True):
        self.train_mode = mode

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        if self.hyper_param.epsilon and self.train_mode and random.random() < self.epsilon.next():
            return 2 * np.random.random_sample((self.num_agents, self.action_size)) - 1
        else:
            return [self.agents[i].act(state[i], add_noise) for i in range(self.num_agents)]

    def reset(self):
        for agent in self.agents:
            agent.reset()
