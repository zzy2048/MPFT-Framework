import torch.nn as nn
import torch
import torch.nn.functional as f
from torch.distributions import Normal


def AvgL1Norm(x, eps=1e-8):
    return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)


class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linear1 = nn.Linear(state_dim, 256)
        self.linear2 = nn.Linear(256, 256)

        self.mean_linear = nn.Linear(256, action_dim)
        self.mean_linear.weight.data.uniform_(-3e-3, 3e-3)
        self.mean_linear.bias.data.uniform_(-3e-3, 3e-3)

        self.log_std_linear = nn.Linear(256, action_dim)
        self.log_std_linear.weight.data.uniform_(-3e-3, 3e-3)
        self.log_std_linear.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        # x = AvgL1Norm(state)
        x = f.relu(self.linear1(state))
        x = f.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, -20, 2)

        return mean, log_std

    def action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        z = normal.sample()
        action = torch.tanh(z).detach().cpu().numpy()

        return action

    # Use re-parameterization tick
    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        noise = Normal(0, 1)

        z = noise.sample().to(self.device)
        action = torch.tanh(mean + std * z)
        log_prob = normal.log_prob(mean + std * z) - torch.log(1 - action.pow(2) + epsilon)

        return action, log_prob.sum(dim=-1).unsqueeze(-1)
