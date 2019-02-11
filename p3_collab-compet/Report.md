[//]: # (Image References)

[rewards]: scores.jpg "Learning Rewards"
[algorithm]: algorithm.jpg "DDPG Algorithm"
[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"


# Project: Continuous Control

## Introduction

For this project, we will work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

## Implementation

The implementation is utilizing DDPG (Deep Deterministic Policy Gradients) with 2 hidden layers for actor & critic networks.
Code is adapted from Udacity's [ddpg-pendulum](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum) repository.


### Actor & Critic architecture

```
Actor(
  (fc1): Linear(in_features=24, out_features=24, bias=True)
  (bn1): BatchNorm1d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc2): Linear(in_features=24, out_features=25, bias=True)
  (bn2): BatchNorm1d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc3): Linear(in_features=25, out_features=2, bias=True)
  (drop_layer): Dropout(p=0.2)
)
Critic(
  (fc1): Linear(in_features=24, out_features=128, bias=True)
  (bn1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc2): Linear(in_features=130, out_features=64, bias=True)
  (bn2): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc3): Linear(in_features=64, out_features=1, bias=True)
  (drop_layer): Dropout(p=0.2)
)
```

### Hyperparameters

Hyperparameters chosen for the implementation are below:

```python
SEED = 0                    # random seed for python, numpy & torch
episodes = 2000             # max episodes to run
max_t = 800                 # max steps in episode
solved_threshold = 0.5      # finish training when avg. score in 100 episodes crosses this threshold

batch_size = 128            # minibatch size
buffer_capacity = int(1e6)  # replay buffer size

n_time_steps_n = 20         # how many steps to collect experiences before learning
n_learn_updates = 10        # how many times to take samples from memory while learning
gamma = 0.999               # discount factor
tau = 2e-2                  # for soft update of target parameters
learning_rate = 1e-4        # learning rate for both actor & critic networks
max_norm = 1                # clipping of gradients to prevent gradient explosion

lr_actor = 3e-3             # learning rate for both actor & critic networks
eps_actor = 1e-8
lr_critic = 5e-3            # learning rate for both actor & critic networks
eps_critic = 1e-8

actor_fc1 = 128             # Actor first hidden layer
actor_fc2 = 130             # Actor second hidden layer
critic_fc1 = 24             # Critic first hidden layer
critic_fc2 = 25             # Critic second hidden layer


```


## Learning Algorithm

Algorithm that was used in this work for solving the enviromnent is described in paper: 
[CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING](https://arxiv.org/pdf/1509.02971.pdf), Timothy P. Lillicrap et al.

![DDPG Algorithm][algorithm]

Because this environment is using two actors, which see local state and perform self-relative actions we can utilize that
knowledge to:
- reuse one actor network and one critic network to double the rate of generating samples and learn those networks more quickly
- share experience buffer. Since values are relative (e.g. distance to the middle) the experience is added from both actors 

Given above we can use any Actor-Critic method to solve this MARL (Multi Agent Reinforcement Learning) enviromnent.
In project 2 DDPG proved to be quite useful so I'll reuse it in this scenario.

Moreover two additional techniques were implemented:
1. **Forced positive experiences relearning** - the experience buffer yields samples which have always 0.25 registered positive values. That way we force quick gains and maybe overfitting at first, but Q-target with tau parameter guards the network from quickly overfitting the positive experience tuples. 
2. **Exploration factor based on experiences** - the random noise is blended into the action response values based on the ratio between positive experiences and non-positive experiences in the buffer. If the ratio (pos/non_pos) < 0.1 then the exploration factor (0.0 -> 1.0) (increases the importance in the response action) is increased linearly. That technique prooved in the traning as good initial bootstrapping of some positive experiences uponn which agent can learn.


 
DDPG is an Actor-Critic method that uses value function and direct policy approximation at the same time.
There are two internal types of neural networks:
- Actor network - transforms state to action values. In this environment there are 4 action values.
- Critic network - transforms state and action values to a quality measure of this state (Q(s, a))


DDPG is using both value function (Critic network) & policy approximation (Actor network) because using only one kind of approximation we get:

- policy based methods - have high variance
- value function methods - have high bias

Moreover, in contrast to DQN we can use continuous action space.

Similarly like for DQN we utilize 

**Experience Replay** 

A buffer with experience tuples (s, a, r, s'): (state, action, reward, next_state)

**Q-targets fixing**

We create 2 neural networks (NN): local and target.
Then fix target NN weights for some learning steps to decouple 
the local NN from target NN parameters making learning more stable and less likely to diverge or fall into oscillations.

Hence in reality we have to have 4 neural networks:
- Critic target NN
- Critic local NN (for execution) 
- Actor target NN 
- Actor local NN (for execution)


## Rewards

As shown below agent learned the environment fairly quickly. In around 1200 episodes it reached average 0.5+ reward. 

![Learning Rewards][rewards]


The weights of the networks are stored in `checkpoint_critic_solved_71.pth`, `checkpoint_actor_solved_71.pth` for critic and actor networks respectively.  

## Ideas for Future Work

**Prioritized Experience Replay**

This approach comes from the idea that we want to focus our training on the actions that were "way off" what we did.
That means the higher the TD error the higher priority. We store in the experience buffer the probability of choosing the experience tuple depending on the TD error. 
Then we sample the experiences based on this probability.
There is one caveat that it's required to add small epsilon to the probability (_p_) since setting the tuple's `p = 0` will make it practically disappear and the agent will never see that tuple again.   

**Trust Region Policy Optimization (TRPO) and Truncated Natural Policy Gradient (TNPG)**

In the trust region, we determine the maximum step size that we want to explore for optimization and then we locate the optimal point within this trust region.
To control the learning speed better, we can be expanded or shrink this trust region in runtime according to the curvature of the surface.
This technique is used because traditional policy based & gradient optimization model for RL might make a too large step and actually fall down (in terms of rewards) and never recover again.
 