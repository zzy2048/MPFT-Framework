import os
import random
from copy import deepcopy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import ppo
import numpy as np
from tqdm import tqdm
import hypervolume

# 设置全局设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# PA2D-MORL主算法类
def is_dominated(obj1, obj2):
    """ obj1 是否被 obj2 支配 """
    return np.all(obj2 >= obj1) and np.any(obj2 > obj1)


class INNER:
    def __init__(self, environment, eval_env, num_objectives, lr=3e-4, seed=42, Xi_k=2000, Psi_k=2000,
                 buffer_size=2048, pareto_ascent_num=2, opposite_num=1, J_max=np.array([3300., 3200.]), use_seed=True):
        self.pareto_ascent_num = pareto_ascent_num
        self.opposite_num = opposite_num

        # Walk2d: 800  || Humanoid: 2000
        self.pareto_front_size = Psi_k

        # Walk2d:  800   || Humanoid: 2000
        self.obj_episodes = Xi_k

        self.max_train_steps = self.obj_episodes * 2048 + 8e5  # 用于控制学习率衰减

        # Walk2d: [3500., 3250.]  || Humanoid [3300., 3200.] || Hopper-3 [2500, 2500, 3200]
        self.reward_max = J_max
        self.seed = seed
        self.lr = lr
        self.env = environment
        self.eval_env = eval_env
        if use_seed:
            np.random.seed(self.seed)
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            self.env.action_space.seed(self.seed)  # 动作空间的种子
        self.agent = ppo.PPO(self.env, self.eval_env, self.seed, num_objectives, max_train_steps=self.max_train_steps,
                             steps=buffer_size)
        self.max_action = float(self.env.action_space.high[0])
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.set_adam_eps = True
        self.use_memory = False
        self.num_objectives = num_objectives
        # 存储策略和对应的目标值 (policy, optimizer, objectives)
        self.non_dominated_set = []
        self.hv_history = []
        self.sp_history = []

    # 添加策略保存方法
    def save_policies(self, obj, save_dir="saved_policies/PPO/"):
        """保存非支配集中的策略到指定目录"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for idx, (policy, _, objectives) in enumerate(self.non_dominated_set):
            # 生成文件名：策略索引+目标值
            obj_str = "_".join([f"{o:.2f}" for o in objectives])
            filename = f"ParetoEdge_{obj}_policy_{idx}_{obj_str}.pth"
            torch.save(policy.state_dict(), os.path.join(save_dir, filename))

    # 添加可视化方法
    def visualize_training(self, save_path="training_metrics"):
        """绘制训练指标变化曲线"""
        if self.num_objectives == 2:
            save_path += f"_pareto_inner_{self.reward_max[0]}_{self.reward_max[1]}.png"
        else:
            save_path += f"_pareto_inner_{self.reward_max[0]}_{self.reward_max[1]}_{self.reward_max[2]}.png"
        print(f"HV: {self.hv_history}")
        print(f"SP: {self.sp_history}")
        plt.figure(figsize=(12, 6))
        # HV指标
        plt.subplot(1, 2, 1)
        plt.plot(self.hv_history, marker='o', linestyle='-', color='b')
        plt.title(f"Inner: Hypervolume (HV) Trend")
        plt.xlabel("Episodes")
        plt.ylabel("HV Value")
        plt.grid(True)
        # SP指标
        plt.subplot(1, 2, 2)
        plt.plot(self.sp_history, marker='o', linestyle='-', color='r')
        plt.title(f"Inner: Spread (SP) Trend")
        plt.xlabel("Episodes")
        plt.ylabel("SP Value")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def visualize_pareto_front(self, save_path="pareto_front"):
        """绘制Pareto前沿"""
        if self.num_objectives == 2:
            save_path += f"_pareto_inner_{self.reward_max[0]}_{self.reward_max[1]}.png"
        else:
            save_path += f"_pareto_inner_{self.reward_max[0]}_{self.reward_max[1]}_{self.reward_max[2]}.png"
        if not self.non_dominated_set:
            return
        objectives = np.array([entry[2] for entry in self.non_dominated_set])
        print(f"Objectives: {list(objectives)}")
        plt.figure(figsize=(8, 6))
        if self.num_objectives == 2:
            plt.scatter(objectives[:, 0], objectives[:, 1], s=50, edgecolors='k')
            plt.title(f"Inner, 2D Pareto Front")
            plt.xlabel("Objective 1")
            plt.ylabel("Objective 2")
        elif self.num_objectives == 3:
            ax = plt.axes(projection='3d')
            ax.scatter3D(objectives[:, 0], objectives[:, 1], objectives[:, 2])
            ax.set_title(f"Inner, 3D Pareto Front")
            ax.set_xlabel("Objective 1")
            ax.set_ylabel("Objective 2")
            ax.set_zlabel("Objective 3")
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

    def update_non_dominated_set(self, policy_optimizers):
        """
        更新非支配解集 self.non_dominated_set 其中元素为：(policy, optimizer, objectives)
        :param policy_optimizers:
        :return:
        """
        new_entries = []
        for policy, optimizer in policy_optimizers:
            new_policy = deepcopy(policy)
            new_optimizer = torch.optim.Adam(new_policy.parameters(), lr=self.lr)
            new_optimizer.load_state_dict(optimizer.state_dict())
            objectives = self.agent.evaluate_policy(new_policy)
            new_policy.to("cpu")
            for param in new_optimizer.state.values():
                for k, v in param.items():
                    if isinstance(v, torch.Tensor):
                        param[k] = v.cpu()
            new_entries.append((new_policy, new_optimizer, objectives))
        # 更新非支配集
        for new_entry in new_entries:
            to_remove = []
            add_flag = True
            for idx, entry in enumerate(self.non_dominated_set):
                if is_dominated(new_entry[2], entry[2]):
                    add_flag = False
                    break
                elif is_dominated(entry[2], new_entry[2]):
                    to_remove.append(idx)
            # 删除被支配的旧策略
            for idx in reversed(to_remove):
                del self.non_dominated_set[idx]
            if add_flag:
                self.non_dominated_set.append(new_entry)
        print(f" INNER, pareto policies size: {len(self.non_dominated_set)}")

    def HV(self):
        if not self.non_dominated_set:
            return 0.0
        objs = np.array([entry[2] for entry in self.non_dominated_set])
        ref_point = np.zeros((self.num_objectives,))
        HV = hypervolume.InnerHyperVolume(ref_point)
        return HV.compute(objs)

    def SP(self):
        if len(self.non_dominated_set) < 2:
            return 0.0
        objs = np.array([entry[2] for entry in self.non_dominated_set])
        sp = 0.0
        for i in range(self.num_objectives):
            sorted_obj = np.sort(objs[:, i])
            sp += np.sum((sorted_obj[1:] - sorted_obj[:-1]) ** 2)
        return sp / (len(objs) - 1)

    def update_ParetoFront(self, policy, optimizer, obj):
        omega = self.agent.obtain_alpha_star(policy, optimizer, obj_id=obj)  # 不包含目标 obj_id 的 Pareto 上升方向
        print(f" ++++omega1: {list(omega)}")
        for _ in range(self.opposite_num):
            self.agent.update(policy, optimizer, omega)
            self.update_non_dominated_set([(policy, optimizer)])
        omega = self.agent.obtain_alpha_star(policy, optimizer)  # 包含 obj_id 的 Pareto 上升方向
        print(f" ++++omega2: {list(omega)}")
        for _ in range(self.pareto_ascent_num):
            self.agent.update(policy, optimizer, omega)
            self.update_non_dominated_set([(policy, optimizer)])

    # ---------------------------> 训练 <-----------------------------
    def train(self):
        with tqdm(total=self.obj_episodes + self.pareto_front_size, desc="INNER Training") as pbar:
            policy = ppo.Policy_Value(self.state_dim, self.action_dim, self.num_objectives, max_action=self.max_action)
            if self.set_adam_eps:
                optimizer = torch.optim.Adam(policy.parameters(), lr=self.lr, eps=1e-5)
            else:
                optimizer = torch.optim.Adam(policy.parameters(), lr=self.lr)
            if self.use_memory:
                optimal_policy = ppo.Policy_Value(self.state_dim, self.action_dim, self.num_objectives,
                                                  max_action=self.max_action)
            optimal_rewards = 0
            flag = True
            # 寻找Inner策略
            for i in range(self.obj_episodes):
                rewards = self.agent.evaluate_policy(policy)
                rewards = self.reward_max / rewards
                omega = ppo.getOmega(rewards)
                if self.use_memory:
                    weighted_reward = np.sum(rewards * omega)
                    if weighted_reward >= optimal_rewards:
                        optimal_rewards = weighted_reward
                        ppo.updateOptimal(policy, optimal_policy)
                        flag = False
                    if i % 100 == 0:
                        if flag:
                            ppo.memoryPolicy(policy, optimal_policy, flag)
                        else:
                            flag = True
                if i > self.obj_episodes / 2:
                    self.update_non_dominated_set([(policy, optimizer)])
                print(f"========== INNER, Episodes {i}, omega {list(omega)} =========")
                self.agent.update(policy, optimizer, omega)
                pbar.update(1)
            # Pareto front 在线追踪
            for p in optimizer.param_groups:
                p['lr'] = self.lr
            self.agent.pareto_tracking = True
            for obj in range(self.num_objectives):
                print(f"========== 沿着目标 {obj} 的反方向搜索")
                new_policy = deepcopy(policy)
                new_optimizer = torch.optim.Adam(new_policy.parameters(), lr=self.lr)
                new_optimizer.load_state_dict(optimizer.state_dict())
                for i in range(int(self.pareto_front_size // self.num_objectives)):
                    self.update_ParetoFront(new_policy, new_optimizer, obj)
                    if i % 10 == 0:
                        current_hv = self.HV()
                        current_sp = self.SP()
                        self.hv_history.append(current_hv)
                        self.sp_history.append(current_sp)
                        print(f"INNER | HV:{current_hv}, SP:{current_sp}")
                    pbar.update(1)
            self.visualize_training()
            self.visualize_pareto_front()
