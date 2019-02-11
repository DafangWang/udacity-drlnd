class HyperParam:
    def __init__(self):
        self.epsilon = False
        self.actor_fc1 = 128
        self.actor_fc2 = 128
        self.critic_fc1 = 128
        self.critic_fc2 = 128
        self.lr_actor = 1e-3
        self.lr_critic = 1e-3
        self.eps_actor = 1e-7
        self.eps_critic = 1e-7
        self.tau = 1e-4
        self.buffer_size = int(1e6)
        self.batch_size = 128
        self.n_learn_updates = 10
        self.n_time_steps = 20
        self.gamma = 0.99
