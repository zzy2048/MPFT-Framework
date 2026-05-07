import torch.nn as nn
import torch
import torch.nn.functional as f


class ValueNet(nn.Module):
    def __init__(self, state_dim, num_obj):
        super(ValueNet, self).__init__()
        self.linear1 = nn.Linear(state_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, num_obj)

        self.linear3.weight.data.uniform_(-3e-3, 3e-3)
        self.linear3.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = f.relu(self.linear1(state))
        x = f.relu(self.linear2(x))
        x = self.linear3(x)
        return x  # 输出形状：(batch_size, num_obj)
