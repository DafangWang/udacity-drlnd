[//]: # (Image References)

[learning_scores]: learning_scores.jpg "Learning Scores"
[learning_scores_list]: learning_scores_list.jpg "Learning Scores List"
[dqn_architecture]: dqn_architecture.jpg "DQN Architecture"


# Project: Navigtation

### Implementation

The implementation is utilizing Deep Q-Network with 3 hidden layers (500, 200, 100 nodes respectively).
Hyperparameters chosen for the implementation are below:

```python
BUFFER_SIZE = int(5e4)  # replay buffer size
BATCH_SIZE = 32         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
SEED = 42
EPISODES = 2000
MAX_TIMESTEPS = int(1e3)
hidden_layers = [500, 200, 100]
```

### Learning Algorithm

This Deep Q-Network learning algorithm uses those steps:

1. Sample environment and store experience tuples **(s, a, r, s')** and store it in replay buffer with some size
2. Once the experience buffer has enough tuples and certain threshold is crossed - get random minibatch from experience buffer and train the network on those tuples
3. While training, fix the target weights for *C* learning steps to make the algoritm more stable and update set of weights using gradient descent.


![DQN architecture][dqn_architecture]


### Rewards

As shown below agent learned the environment fairly quickly. In around 500 episodes it reached average 13+ reward. 

![Learning Scores List][learning_scores_list]

![Learning Scores Chart][learning_scores]

The weights of the networks are stored in `checkpoint.pth` file using `torch.save(agent.local.state_dict(), 'checkpoint.pth')`

### Ideas for Future Work

The Deep Q-Network can be further improved by for example:

- Modifying replay buffer to make the experience tuples prioritized: the higher the TD error the higher priority
- Using Double DQN - one set of weights for selecting best action and other one to evaluate it
- Using Dueling DQN (DDQN) - Introduce so called advantage values **A(s, a)** - the advantage of taking the action at the state (how much better is it to take this action vs other). Then use one network to estimate state values and the other that estimates advantages **A(s, a)**. From both of those comptre Q values.