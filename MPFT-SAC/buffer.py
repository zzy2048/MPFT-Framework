import random

import numpy as np
import torch
import collections


class ReplayBuffer:
    def __init__(self, buffer_maxLen=1e6):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.buffer_maxLen = int(buffer_maxLen)
        self.batch_size = 256
        self.buffer = collections.deque(maxlen=self.buffer_maxLen)

    def push(self, data):
        self.buffer.append(data)

    def sample(self):
        state_list = []
        action_list = []
        reward_list = []
        next_state_list = []
        not_done_list = []
        batch = random.sample(self.buffer, self.batch_size)
        for experience in batch:
            s, a, n_s, r, d = experience
            state_list.append(s)
            action_list.append(a)
            next_state_list.append(n_s)
            reward_list.append(r)
            not_done_list.append(1-d)

        return torch.FloatTensor(np.array(state_list)).to(self.device), \
               torch.FloatTensor(np.array(action_list)).to(self.device), \
               torch.FloatTensor(np.array(reward_list)).to(self.device), \
               torch.FloatTensor(np.array(next_state_list)).to(self.device), \
               torch.FloatTensor(np.array(not_done_list)).unsqueeze(-1).to(self.device)

    def buffer_len(self):
        return len(self.buffer)

    def clear_buffer(self):
        self.buffer = collections.deque(maxlen=self.buffer_maxLen)
