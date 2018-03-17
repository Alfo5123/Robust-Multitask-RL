import random
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'time'))

class ReplayMemory(object):

    def __init__(self, capacity, policy_capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

        self.policy_capacity = policy_capacity
        self.policy_memory = []
        self.policy_position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

        if len(self.policy_memory) < self.policy_capacity:
            self.policy_memory.append(None)
        self.policy_memory[self.policy_position] = Transition(*args)
        self.policy_position = (self.policy_position + 1) % self.policy_capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def policy_sample(self, batch_size):
        return random.sample(self.policy_memory, batch_size)

    def __len__(self):
        return len(self.memory)
