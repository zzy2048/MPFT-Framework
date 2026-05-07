import torch
import torch.optim as optim
import numpy as np
import buffer
import PolicyNet
import qNet
import valueNet
import torch.nn as nn
import paretoAscent
from normalization import Normalization, RewardScaling


def copy_from(agent, new_agent):
    # 复制神经网络参数
    new_agent.value_net.load_state_dict(agent.value_net.state_dict())
    new_agent.target_value_net.load_state_dict(agent.target_value_net.state_dict())
    new_agent.q1_net.load_state_dict(agent.q1_net.state_dict())
    new_agent.q2_net.load_state_dict(agent.q2_net.state_dict())
    new_agent.policy_net.load_state_dict(agent.policy_net.state_dict())

    # 复制log_alpha并确保设备正确
    with torch.no_grad():
        new_agent.log_alpha.copy_(agent.log_alpha.to(new_agent.device))
    new_agent.alpha = new_agent.log_alpha.exp().detach()

    # 复制优化器状态
    new_agent.value_optimizer.load_state_dict(agent.value_optimizer.state_dict())
    new_agent.q1_optimizer.load_state_dict(agent.q1_optimizer.state_dict())
    new_agent.q2_optimizer.load_state_dict(agent.q2_optimizer.state_dict())
    new_agent.policy_optimizer.load_state_dict(agent.policy_optimizer.state_dict())
    new_agent.alpha_optim.load_state_dict(agent.alpha_optim.state_dict())

    # 复制训练状态变量
    new_agent.training_steps = agent.training_steps
    new_agent.omega = agent.omega.copy()  # numpy数组深拷贝
    new_agent.grad_normalize = agent.grad_normalize  # 布尔值直接赋值


def getOmega(rewards):
    """
        根据奖励 rewards 中值的大小调整权重 omega
        rewards[i] 越大 omega[i] 越小
     """
    if (rewards < 0).any():
        rewards = np.ones_like(rewards)
    return rewards / np.sum(rewards)


class SAC:
    def __init__(self, state_dim, action_dim, num_obj, rl, max_action=1):
        self.tau = 0.001
        self.gamma = 0.99
        self.num_obj = num_obj
        self.lr = rl
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training_steps = 0
        self.omega = np.ones((self.num_obj,)) / self.num_obj
        self.grad_normalize = False
        self.state_normalize = False
        self.reward_scaling = True
        self.state_norm = Normalization(shape=state_dim)
        self.reward_scal = RewardScaling(shape=num_obj, gamma=self.gamma)
        self.PA = paretoAscent.ParetoAscentDirection()
        self.max_action = max_action
        # Temperature
        self.log_alpha = torch.zeros(num_obj).to(self.device)
        self.alpha = self.log_alpha.exp().to(self.device)
        self.target_entropy = [-action_dim]  # -|A|
        self.target_entropy = torch.FloatTensor(self.target_entropy).to(self.device)
        # initialize networks
        self.value_net = valueNet.ValueNet(state_dim=state_dim, num_obj=num_obj).to(self.device)
        self.target_value_net = valueNet.ValueNet(state_dim=state_dim, num_obj=num_obj).to(self.device)
        self.q1_net = qNet.QNet(state_dim=state_dim, action_dim=action_dim, num_obj=num_obj).to(self.device)
        self.q2_net = qNet.QNet(state_dim=state_dim, action_dim=action_dim, num_obj=num_obj).to(self.device)
        self.policy_net = PolicyNet.PolicyNet(state_dim=state_dim, action_dim=action_dim).to(
            self.device)

        # Load the target value network parameters
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        # Initialize the optimizer
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.lr)
        self.q1_optimizer = optim.Adam(self.q1_net.parameters(), lr=self.lr)
        self.q2_optimizer = optim.Adam(self.q2_net.parameters(), lr=self.lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.log_alpha.requires_grad = True
        self.alpha.requires_grad = True
        self.alpha_optim = optim.Adam([self.log_alpha], lr=self.lr)
        # Initialize the buffer
        self.buffer = buffer.ReplayBuffer()

    def evaluate_policy(self, eval_env, seed, eval_episodes=2):
        total_returns = np.zeros(self.num_obj)
        t = 0
        for i in range(eval_episodes):
            state = eval_env.reset(seed=seed + i + 1)
            done = False
            returns = np.zeros((self.num_obj,))
            discount = 1
            while not done:
                if self.state_normalize:
                    state = self.state_norm(state, update=False)  # During the evaluating,update=False
                with torch.no_grad():
                    action = self.get_action(state)
                next_state, reward, done, _, _ = eval_env.step(action)
                returns += discount * reward
                # discount *= self.gamma
                state = next_state
                t += 1
            total_returns += returns
        total_returns = total_returns / eval_episodes
        print(f"Rewards: {list(total_returns)}, steps:{t // eval_episodes} ")
        return total_returns

    def get_action(self, state):
        action = self.policy_net.action(state)
        return action

    # omega 为权重
    def update(self, omega=None, obj_id=-1, update_info=""):
        if self.buffer.buffer_len() < 256:
            return
        self.training_steps += 1
        state, action, reward, next_state, not_done = self.buffer.sample()
        new_action, log_prob = self.policy_net.evaluate(state)

        # V value loss
        value = self.value_net(state)
        new_q1_value = self.q1_net(state, new_action)
        new_q2_value = self.q2_net(state, new_action)
        q_value = torch.min(new_q1_value, new_q2_value)
        next_value = q_value - log_prob * self.alpha
        value_loss = nn.MSELoss()(value, next_value.detach())

        # Soft q  loss
        q1_value = self.q1_net(state, action)
        q2_value = self.q2_net(state, action)
        target_value = self.target_value_net(next_state)
        target_q_value = reward + not_done * self.gamma * target_value
        q1_value_loss = nn.MSELoss()(q1_value, target_q_value.detach())
        q2_value_loss = nn.MSELoss()(q2_value, target_q_value.detach())

        # 计算 omega
        if omega is None:
            if self.training_steps % 2000 == 0:  # 每一个Episode更新一次
                omega = self.obtain_alpha_star(log_prob, q_value, obj_id)
                self.omega = omega
                print(f"{update_info}: {omega}")
            else:
                omega = self.omega

        # policy loss
        policy_loss = torch.stack([(self.alpha[i] * log_prob - q_value[:, i]).mean() for i in range(self.num_obj)])
        policy_loss = (policy_loss * torch.FloatTensor(omega).to(self.device)).sum()
        alpha_loss = -torch.mean(self.log_alpha * (self.target_entropy + log_prob).detach())

        # Update Policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update v
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Update Soft q
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        q1_value_loss.backward()
        q2_value_loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        # Update temperature alpha
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

        # Update target networks
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

    def solve_pareto_weights(self, grads):
        if self.grad_normalize:
            for i in range(self.num_obj):
                grads[i] /= np.linalg.norm(grads[i])
        alpha = self.PA.solve(np.array(grads))
        return alpha

    def obtain_alpha_star(self, log_prob, q_value, obj_id=-1):
        if obj_id != -1 and (self.num_obj == 2):
            alpha_star = np.zeros((2,))
            alpha_star[obj_id] = 1
            return 1 - alpha_star
        grads = []
        for i in range(self.num_obj):
            if i == obj_id:  # 跳过目标 i
                continue
            policy_loss = (self.alpha[i] * log_prob - q_value[:, i]).mean()
            # 前向计算（仅策略网络）
            self.policy_optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            grad = []
            for param in self.policy_net.parameters():
                if param.grad is not None:
                    grad.append(param.grad.view(-1))
            grads.append(torch.cat(grad).detach().cpu().numpy())

        # 计算 alpha_star
        if self.num_obj == 2 or obj_id != -1:
            grad1, grad2 = grads
            # 正则化，避免优化方向由较小模的梯度主导
            if self.grad_normalize:
                grad1 /= np.linalg.norm(grad1)
                grad2 /= np.linalg.norm(grad2)
            numerator = np.dot(grad2 - grad1, grad2)
            denominator = np.linalg.norm(grad1 - grad2) ** 2 + 1e-8
            alpha1 = max(0.0, min(1.0, numerator / denominator))
            alpha_star = np.array([alpha1, 1 - alpha1])
            if self.num_obj == 3:
                if obj_id == 0:
                    alpha_star = np.array([0, alpha1, 1 - alpha1])
                elif obj_id == 1:
                    alpha_star = np.array([alpha1, 0, 1 - alpha1])
                else:
                    alpha_star = np.array([alpha1, 1 - alpha1, 0])
        else:
            alpha_star = self.solve_pareto_weights(grads)
        return alpha_star

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path + 'SAC_policy.pth')

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path + 'SAC_policy.pth'))
