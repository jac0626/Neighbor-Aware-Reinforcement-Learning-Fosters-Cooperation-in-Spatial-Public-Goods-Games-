import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random
from typing import Tuple

class SharedDQN(nn.Module):
    """
    Shared Deep Q-Network (The 'Social Brain')
    Input: Continuous state vector (batch_size, input_dim)
    Output: Action Q-values (batch_size, 2) -> [Q_coop, Q_defect]
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 2):
        super(SharedDQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    """
    Global Experience Replay Buffer, stores (s, a, r, s') for all agents
    """
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = zip(*batch)
        return (
            torch.FloatTensor(np.array(state)),
            torch.LongTensor(np.array(action)),
            torch.FloatTensor(np.array(reward)),
            torch.FloatTensor(np.array(next_state))
        )

    def __len__(self):
        return len(self.buffer)
