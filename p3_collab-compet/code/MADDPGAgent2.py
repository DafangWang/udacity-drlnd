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

GAMMA = 0.99
WEIGHT_DECAY = 0
EXPLORE_EXPLOIT_DECAY = 0.999

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MADDPGAgent2():
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
            hyper_param.actor_fc1 = 128
            hyper_param.actor_fc2 = 128
            hyper_param.critic_fc1 = 128
            hyper_param.critic_fc2 = 128
            hyper_param.lr_actor = 1e-3
            hyper_param.lr_critic = 1e-3
            hyper_param.tau = 1e-4
            hyper_param.batch_size = 128
            hyper_param.n_learn_updates = 10
            hyper_param.n_time_steps = 20

        self.hyper_param = hyper_param

        self.device = device
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, hyper_param.batch_size, random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed, hyper_param.actor_fc1, hyper_param.actor_fc2).to(
            device)
        self.actor_target = Actor(state_size, action_size, random_seed, hyper_param.actor_fc1,
                                  hyper_param.actor_fc2).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=hyper_param.lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size * num_agents, action_size * num_agents, random_seed).to(device)
        self.critic_target = Critic(state_size * num_agents, action_size * num_agents, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=hyper_param.lr_critic, weight_decay=WEIGHT_DECAY)

        # self.epsilon = SpacedRepetitionDecay(ExponentialDecay(1.0, 0.0, hyper_param.epsilon_decay),
        #                                      hyper_param.epsilon_spaced_init, hyper_param.epsilon_spaced_decay)


        self.epsilon = PositiveMemoriesFactorExplorationDecay(0.5, 0, 0.0002, 0.12, self.memory)

        self.noise = OUNoise(action_size, random_seed)

        self.train_mode = True

        self.actor_loss = []
        self.critic_loss = []

    def train(self, mode=True):
        self.train_mode = mode

    def step(self, time_step, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.memory.add(np.reshape(state, (self.state_size * self.num_agents,)),
                        np.reshape(action, (self.action_size * self.num_agents,)),
                        np.reshape(reward, (self.num_agents,)),
                        np.reshape(next_state, (self.state_size * self.num_agents,)),
                        np.reshape(done, (self.num_agents,)))

        # if time_step % self.hyper_param.n_time_steps != 0:
        #     return

        if len(self.memory) > self.hyper_param.batch_size:
            # for i in range(self.hyper_param.n_learn_updates):
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def _act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        if self.hyper_param.epsilon and self.train_mode and random.random() < self.epsilon.next():
            return 2 * np.random.random_sample(self.action_size) - 1

        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
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
        states_sum, actions_sum, rewards_sum, next_states_sum, dones_sum = experiences

        batch_size = len(states_sum)

        states = np.reshape(states_sum, (self.num_agents, batch_size, -1))
        actions = np.reshape(actions_sum, (self.num_agents, batch_size, -1))
        rewards = np.reshape(rewards_sum, (self.num_agents, batch_size, -1))
        next_states = np.reshape(next_states_sum, (self.num_agents, batch_size, -1))
        dones = np.reshape(dones_sum, (self.num_agents, batch_size, -1))

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = [self.actor_target(next_states[i]) for i in range(self.num_agents)]
        actions_next = torch.cat(actions_next, dim=1)
        actions_next_sum = actions_next.view((batch_size, self.action_size * self.num_agents))

        Q_targets_next_sum = self.critic_target(next_states_sum, actions_next_sum)

        # Compute Q targets for current states (y_i)
        Q_targets_sum = rewards_sum + (gamma * Q_targets_next_sum * (1 - dones_sum))
        # Compute critic loss
        Q_expected_sum = self.critic_local(states_sum, actions_sum)
        critic_loss = F.mse_loss(Q_targets_sum, Q_expected_sum)
        self.critic_loss.append(critic_loss.cpu().data.numpy())

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = [self.actor_local(states[i]) for i in range(self.num_agents)]
        actions_pred = torch.cat(actions_pred, dim=1)
        actions_pred_sum = actions_pred.view((batch_size, self.action_size * self.num_agents))
        actor_loss = -self.critic_local(states_sum, actions_pred_sum).mean()
        self.actor_loss.append(actor_loss.cpu().data.numpy())

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.hyper_param.tau)
        self.soft_update(self.actor_local, self.actor_target, self.hyper_param.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
