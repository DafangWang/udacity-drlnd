import numpy as np


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, state_size, action_size, buffer_capacity):
        self.action_size = action_size
        self.state_size = state_size
        self.buffer_capacity = buffer_capacity

        self.states = self._create_buffer(state_size)
        self.actions = self._create_buffer(action_size)
        self.rewards = self._create_buffer(1)
        self.next_states = self._create_buffer(state_size)
        self.dones = self._create_buffer(1)

        self.index = 0
        self.full = False

    def _create_buffer(self, size):
        return np.empty((self.buffer_capacity, size))

    def _next_index(self):
        self.index += 1
        if self.index >= self.buffer_capacity:
            self.full = True
            self.index = 0

    def size(self):
        if self.full:
            return self.buffer_capacity
        return self.index

    def add(self, state, action, reward, next_state, done):
        self.states[self.index, :] = state
        self.actions[self.index, :] = action
        self.rewards[self.index, :] = [reward]
        self.next_states[self.index, :] = next_state
        self.dones[self.index, :] = [done]

        self._next_index()

    def _get_sample_indices(self, n, exact_batch_size=False):
        samples_to_draw = n
        buffer_size = self.size()

        if samples_to_draw > buffer_size:
            message = "Not enough samples to draw from the ReplayBuffer [size: {}, to_draw: {}]".format(buffer_size,
                                                                                                        samples_to_draw)
            raise Exception(message)

        if samples_to_draw == buffer_size:
            return np.random.permutation(buffer_size)

        if not exact_batch_size:
            return np.random.randint(0, buffer_size, samples_to_draw)

        indices = np.array([])
        while samples_to_draw > 0:
            draw = np.random.random_integers(0, buffer_size, samples_to_draw)
            indices = np.unique(indices, draw)

            samples_to_draw = self.batch_size - len(indices)

        return indices

    def sample(self, n, exact_batch_size=False):
        indices = self._get_sample_indices(n, exact_batch_size)

        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices, 0]
        next_states = self.next_states[indices]
        dones = self.dones[indices, 0]

        return states, actions, rewards, next_states, dones
