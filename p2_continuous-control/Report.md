[//]: # (Image References)

[learning_scores]: learning_scores.jpg "Learning Scores"
[learning_scores_list]: learning_scores_list.jpg "Learning Scores List"
[algorithm]: algorithm.jpg "DDPG Algorithm"
[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"


# Project: Continuous Control

## Introduction

For this project, we will work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

## Implementation

The implementation is utilizing DDPG (Deep Deterministic Policy Gradients) with 2 hidden layers (256, 128 nodes) for both actor & critic networks.
Code is adapted from Udacity's [ddpg-pendulum](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum) repository.

Hyperparameters chosen for the implementation are below:

```python
SEED = 16                   # random seed for python, numpy & torch
episodes = 1000             # max episodes to run
max_t = 1000                # max steps in episode
solved_threshold = 30       # finish training when avg. score in 100 episodes crosses this threshold

batch_size = 128            # minibatch size
buffer_capacity = int(1e6)  # replay buffer size

learn_every_n = 20          # how many steps to collect experiences before learning
learn_updates = 10          # how many times to take samples from memory while learning
gamma = 0.99                # discount factor
tau = 1e-3                  # for soft update of target parameters
learning_rate = 1e-4        # learning rate for both actor & critic networks
max_norm = 1                # clipping of gradients to prevent gradient explosion
```

## Learning Algorithm

Algorithm that was used in this work for solving the enviromnent is described in paper: 
[CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING](https://arxiv.org/pdf/1509.02971.pdf), Timothy P. Lillicrap et al.

![DDPG Algorithm][algorithm]

DDPG is an Actor-Critic method that uses value function and direct policy approximation at the same time.
There are two internal types of neural networks:
- Actor network - transforms state to action values. In this environment there are 4 action values.
- Critic network - transforms state and action values to a quality measure of this state (Q(s, a))


DDPG is using both value function (Critic network) & policy approximation (Actor network) because using only one kind of approximation we get:

- policy based methods - have high variance
- value function methods - have high bias

Moreover, in contrast to DQN we can use continuous action space.

## Rewards

As shown below agent learned the environment fairly quickly. In around 500 episodes it reached average 13+ reward. 

![Learning Scores Chart][learning_scores]

![Learning Scores List][learning_scores_list]

The weights of the networks are stored in `checkpoint.pth` file using `torch.save(agent.local.state_dict(), 'checkpoint.pth')`

## Ideas for Future Work

Prioritized Experience Replay
PER is exactly what it sounds like. Instead of sampling randomly from the experience buffer, as is done in this project, PER assigns an "error" score to each experience based on the difference between the expected reward and the observed reward for that experience. This has not been implemented but could have a significant benefit in reducing the number of episodes required to solve this environment.

Model-based algorithms instead of DDPG
In contrast with model-free RL algorithms that attempt to predict optimal action-value functions or policy functions, model-based RL algorithms go one level deeper to attempt prediction of state transitions given a current state and taking some action. Because of the simplicity underlying the target dynamics in the Reacher environments, it would make sense to try to identify the time-dependent pattern of the target itself.

This approach would open up the realm of time-series-based supervised-learning techniques including RNNs and LSTMs and be expected to provide a much more efficient learning algorithm.