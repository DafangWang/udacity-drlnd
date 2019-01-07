import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.utils as tutils
from torch import optim

from LinearNet import LinearNet
from OUNoise import OUNoise
from TorchReplayBuffer import TorchReplayBuffer


class DDPGAgent:
    def __init__(self, state_size, action_size, config):

        self.state_size = state_size
        self.action_size = action_size
        self.config = config

        self.batch_size = 128
        self.buffer_capacity = 1000000

        self.step_count = 0
        self.min_observations = 20
        self.learn_updates = 10
        self.gamma = 0.99
        self.tau = 1e-3
        self.learning_rate = 1e-4
        self.max_norm = 1

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.actor_local = LinearNet(state_size, [256, 128], action_size).to(self.device)
        self.actor_target = LinearNet(state_size, [256, 128], action_size).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.learning_rate)

        self.critic_local = LinearNet(state_size + action_size, [256, 128], 128).to(self.device)
        self.critic_target = LinearNet(state_size + action_size, [256, 128], 128).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.learning_rate)

        self.noise = OUNoise(action_size)

        self.memory = TorchReplayBuffer(state_size, action_size, self.buffer_capacity, self.device)
        # self.memory = ReplayBuffer(state_size, action_size, self.buffer_capacity)

    def reset(self):
        self.noise.reset()

    def act(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            # TODO: cpu().data.numpy???
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        action += self.noise.sample()
        return np.clip(action, -1, 1)

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.step_count += 1
        if self.step_count % self.min_observations != 0:
            return

        if self.memory.size() > self.batch_size:
            self._learn()

    def _learn(self):
        for _ in range(self.learn_updates):
            experiences = self.memory.sample(self.batch_size)
            self._learn_step(experiences)

    def _learn_step(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        next_actions = self.actor_target(next_states)
        Q_targets_next = self.critic_target(torch.cat((next_states, next_actions), 1))
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        Q_expected = self.critic_local(torch.cat((states, actions), 1))
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        tutils.clip_grad_norm_(self.critic_local.parameters(), self.max_norm)
        self.critic_optimizer.step()

        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(torch.cat((states, actions_pred), 1)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        tutils.clip_grad_norm_(self.actor_local.parameters(), self.max_norm)
        self.actor_optimizer.step()

        self._soft_update(self.critic_local, self.critic_target)
        self._soft_update(self.actor_local, self.actor_target)

    def _soft_update(self, source, destination):
        for dst, src in zip(destination.parameters(), source.parameters()):
            dst.detach_()
            dst.copy_(dst * (1.0 - self.tau) + src * self.tau)
    #
    # def eval_step(self, state):
    #     self.config.state_normalizer.set_read_only()
    #     state = self.config.state_normalizer(state)
    #     action = self.network(state)
    #     self.config.state_normalizer.unset_read_only()
    #     return to_np(action)
    #
    # def step(self):
    #     config = self.config
    #     if self.state is None:
    #         self.random_process.reset_states()
    #         self.state = self.task.reset()
    #         self.state = config.state_normalizer(self.state)
    #     action = self.network(self.state)
    #     action = to_np(action)
    #     action += self.random_process.sample()
    #     next_state, reward, done, _ = self.task.step(action)
    #     next_state = self.config.state_normalizer(next_state)
    #     self.episode_reward += reward[0]
    #     reward = self.config.reward_normalizer(reward)
    #     self.replay.feed([self.state, action, reward, next_state, done.astype(np.uint8)])
    #     if done[0]:
    #         self.episode_rewards.append(self.episode_reward)
    #         self.episode_reward = 0
    #         self.random_process.reset_states()
    #     self.state = next_state
    #     self.total_steps += 1
    #
    #     if self.replay.size() >= config.min_memory_size:
    #         experiences = self.replay.sample()
    #         states, actions, rewards, next_states, terminals = experiences
    #         states = states.squeeze(1)
    #         actions = actions.squeeze(1)
    #         rewards = tensor(rewards)
    #         next_states = next_states.squeeze(1)
    #         terminals = tensor(terminals)
    #
    #         phi_next = self.target_network.feature(next_states)
    #         a_next = self.target_network.actor(phi_next)
    #         q_next = self.target_network.critic(phi_next, a_next)
    #         q_next = config.discount * q_next * (1 - terminals)
    #         q_next.add_(rewards)
    #         q_next = q_next.detach()
    #         phi = self.network.feature(states)
    #         q = self.network.critic(phi, tensor(actions))
    #         critic_loss = (q - q_next).pow(2).mul(0.5).sum(-1).mean()
    #
    #         self.network.zero_grad()
    #         critic_loss.backward()
    #         self.network.critic_opt.step()
    #
    #         phi = self.network.feature(states)
    #         action = self.network.actor(phi)
    #         policy_loss = -self.network.critic(phi.detach(), action).mean()
    #
    #         self.network.zero_grad()
    #         policy_loss.backward()
    #         self.network.actor_opt.step()
    #
    #         self.soft_update(self.target_network, self.network)
