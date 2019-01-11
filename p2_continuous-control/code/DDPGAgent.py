import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from LinearNet import Actor, Critic
from OUNoise import OUNoise
from ReplayBuffer import ReplayBuffer
from TorchReplayBuffer import TorchReplayBuffer


class DDPGAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.batch_size = 128           # minibatch size
        self.buffer_capacity = int(1e6) # replay buffer size

        self.learn_every_n = 20         # how many steps to collect experiences before learning
        self.learn_updates = 10         # how many times to take samples from memory while learning
        self.gamma = 0.99               # discount factor
        self.tau = 1e-3                 # for soft update of target parameters
        self.learning_rate = 1e-4       # learning rate for both actor & critic networks
        self.max_norm = 1               # clipping of gradients to prevent gradient explosion

        self.step_count = 0

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.actor_local = Actor(state_size, action_size).to(self.device)
        self.actor_target = Actor(state_size, action_size).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.learning_rate)

        self.critic_local = Critic(state_size, action_size).to(self.device)
        self.critic_target = Critic(state_size, action_size).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.learning_rate)

        self.noise = OUNoise(action_size)

        self.memory = TorchReplayBuffer(ReplayBuffer(state_size, action_size, self.buffer_capacity), self.device)

    def reset(self):
        self.noise.reset()

    def act(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        action += self.noise.sample()
        action = np.clip(action, -1, 1)
        return action

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.step_count = (self.step_count + 1) % self.learn_every_n
        if self.step_count == 0 and self.memory.size() > self.batch_size:
            self._learn()

    def _learn(self):
        for _ in range(self.learn_updates):
            experiences = self.memory.sample(self.batch_size)
            self._learn_step(experiences)

    def _combine(self, state, actions):
        return torch.cat((state, actions), 1)

    def _learn_step(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        next_actions = self.actor_target(next_states)
        # Q_targets_next = self.critic_target(self._combine(next_states, next_actions))
        Q_targets_next = self.critic_target(next_states, next_actions)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Q_expected = self.critic_local(self._combine(states, actions))
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), self.max_norm)
        self.critic_optimizer.step()

        actions_pred = self.actor_local(states)
        # actor_loss = -self.critic_local(self._combine(states, actions_pred)).mean()
        actor_loss = -self.critic_local(states, actions_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), self.max_norm)
        self.actor_optimizer.step()

        self._soft_update(self.critic_local, self.critic_target)
        self._soft_update(self.actor_local, self.actor_target)

    def _soft_update(self, source, destination):
        for dst, src in zip(destination.parameters(), source.parameters()):
            dst.detach_()
            dst.copy_(dst * (1.0 - self.tau) + src * self.tau)
