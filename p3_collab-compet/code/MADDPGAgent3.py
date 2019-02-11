import copy
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from DDPGAgent import ReplayBuffer, OUNoise
from DecayingValue import ExponentialDecay, SpacedRepetitionDecay, PositiveMemoriesFactorExplorationDecay
from HyperParam import HyperParam
from LinearNet import Actor, Critic

BUFFER_SIZE = int(1e6)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MADDPGAgent3():
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
            hyper_param.epsilon = False
            hyper_param.actor_fc1 = 128
            hyper_param.actor_fc2 = 128
            hyper_param.critic_fc1 = 128
            hyper_param.critic_fc2 = 128
            hyper_param.lr_actor = 1e-3
            hyper_param.lr_critic = 1e-3
            hyper_param.eps_actor = 1e-7
            hyper_param.eps_critic = 1e-7
            hyper_param.tau = 1e-4
            hyper_param.buffer_size = int(1e6)
            hyper_param.batch_size = 128
            hyper_param.n_learn_updates = 10
            hyper_param.n_time_steps = 20
            hyper_param.gamma = 0.99

        self.hyper_param = hyper_param

        self.device = device
        self.memory = ReplayBuffer(action_size, self.hyper_param.buffer_size, self.hyper_param.batch_size, random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed, hyper_param.actor_fc1, hyper_param.actor_fc2).to(
            device)
        self.actor_target = Actor(state_size, action_size, random_seed, hyper_param.actor_fc1,
                                  hyper_param.actor_fc2).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=hyper_param.lr_actor, eps=hyper_param.eps_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=hyper_param.lr_critic, eps=hyper_param.eps_critic)

        self.hard_update(self.actor_target, self.actor_local)
        self.hard_update(self.critic_target, self.critic_local)

        self.noise = OUNoise(action_size, random_seed, mu=0.0)

        self.train_mode = True

        self.actor_loss = []
        self.critic_loss = []

        self.orig_actions = [[0.0, 0.0], [0.0, 0.0]]

        if hyper_param.epsilon:
            self.epsilon = hyper_param.epsilon_model(self.memory)

    def train(self, mode=True):
        self.train_mode = mode

    def step(self, time_step, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        if self.train_mode is False:
            return

        for i in range(self.num_agents):
            self.memory.add(state[i], action[i], reward[i], next_state[i], done[i])

        if time_step % self.hyper_param.n_time_steps != 0:
            return

        if len(self.memory) > self.hyper_param.batch_size:
            for i in range(self.hyper_param.n_learn_updates):
                experiences = self.memory.sample()
                self.learn(experiences, self.hyper_param.gamma)

    def _act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        batch_state = np.reshape(state, (1, self.state_size))
        state = torch.from_numpy(batch_state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        action = np.reshape(action, (self.action_size,))
        self.orig_actions = copy.copy(action)

        if add_noise and self.train_mode:
            eps = self.epsilon.next()
            action = (1.0 - eps) * action + eps * self.noise.sample()

            if self.hyper_param.epsilon:
                random_action = 2 * np.random.random_sample(self.action_size) - 1
                action = (1.0 - eps) * action + eps * random_action

        return np.clip(action, -1, 1)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        return [self._act(state[i], add_noise) for i in range(self.num_agents)]

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.smooth_l1_loss(Q_targets, Q_expected)
        # critic_loss = F.mse_loss(Q_targets, Q_expected)
        self.critic_loss.append(critic_loss.cpu().data.numpy())

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        self.actor_loss.append(actor_loss.cpu().data.numpy())

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 0.5)
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.hyper_param.tau)
        self.soft_update(self.actor_local, self.actor_target, self.hyper_param.tau)


    def soft_update(self, from_model, to_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for to_param, from_param in zip(to_model.parameters(), from_model.parameters()):
            to_param.data.copy_(tau * from_param.data + (1.0 - tau) * to_param.data)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
