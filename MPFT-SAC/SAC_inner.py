import os
import random
from copy import deepcopy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import SAC
import numpy as np
from tqdm import tqdm
from environments import ant, half_cheetah, hopper_2, hopper_3, humanoid, swimmer, walker2d
import hypervolume

# 设置全局设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# PA2D-MORL主算法类
def is_dominated(obj1, obj2):
    """ obj1 是否被 obj2 支配 """
    return np.all(obj2 >= obj1) and np.any(obj2 > obj1)


class SAC_INNER:
    def __init__(self, environment, eval_env, num_objectives, lr=3e-4, seed=42, Xi_k=500, Psi_k=600,
                 steps=2000, pareto_ascent_num=2, opposite_num=1, J_max=np.array([3000., 3500.]), use_seed=False):
        self.pareto_ascent_num = pareto_ascent_num
        self.opposite_num = opposite_num
        # Walk2d: 600  || Humanoid: 1700
        self.pareto_front_size = Psi_k
        # Walk2d:  500   || Humanoid: 1500
        self.obj_episodes = Xi_k
        self.steps = steps  # 每个episode的步长
        self.edge_direction = 0
        # Walk2d: [3000., 3500.]  || Humanoid [5300., 6200.]
        self.reward_max = J_max

        self.clear_buffer = False
        self.seed = seed
        self.lr = lr
        self.env = environment
        self.eval_env = eval_env
        if use_seed:
            np.random.seed(self.seed)
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            self.env.action_space.seed(self.seed)  # 动作空间的种子
            self.eval_env.action_space.seed(self.seed)  # 动作空间的种子
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.num_objectives = num_objectives
        self.max_action = float(self.env.action_space.high[0])
        self.agent = SAC.SAC(self.state_dim, self.action_dim, self.num_objectives, self.lr, max_action=self.max_action)
        # 存储策略和对应的目标值 (policy_net, q_net, optimizer, objectives)
        self.non_dominated_set = []
        self.hv_history = []
        self.sp_history = []

    # 添加策略保存方法
    def save_policies(self, obj, save_dir="saved_policies"):
        """保存非支配集中的策略到指定目录"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx, (policy, q_net, _, objectives) in enumerate(self.non_dominated_set):
            # 生成文件名：策略索引+目标值
            obj_str = "_".join([f"{o:.2f}" for o in objectives])
            filename1 = f"INNER_policy_{idx}_{obj_str}.pth"
            filename2 = f"INNER_Qnet_{idx}_{obj_str}.pth"
            torch.save(policy.state_dict(), os.path.join(save_dir, filename1))
            torch.save(q_net.state_dict(), os.path.join(save_dir, filename2))

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
        plt.title(f"ParetoINNER: Hypervolume (HV) Trend")
        plt.xlabel("Episodes")
        plt.ylabel("HV Value")
        plt.grid(True)
        # SP指标
        plt.subplot(1, 2, 2)
        plt.plot(self.sp_history, marker='o', linestyle='-', color='r')
        plt.title(f"ParetoINNER: Spread (SP) Trend")
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
        objectives = np.array([entry[3] for entry in self.non_dominated_set])
        print(f"Objectives: {list(objectives)}")
        plt.figure(figsize=(8, 6))
        if self.num_objectives == 2:
            plt.scatter(objectives[:, 0], objectives[:, 1], s=50, edgecolors='k')
            plt.title(f"Pareto INNER, 2D Pareto Front")
            plt.xlabel("Objective 1")
            plt.ylabel("Objective 2")
        elif self.num_objectives == 3:
            ax = plt.axes(projection='3d')
            ax.scatter3D(objectives[:, 0], objectives[:, 1], objectives[:, 2])
            ax.set_title(f"Pareto INNER, 3D Pareto Front")
            ax.set_xlabel("Objective 1")
            ax.set_ylabel("Objective 2")
            ax.set_zlabel("Objective 3")
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

    def update_non_dominated_set(self, policy_Qnet_optimizers):
        """
        更新非支配解集 self.non_dominated_set 其中元素为：(policy, optimizer, objectives)
        :param policy_Qnet_optimizers:
        :return:
        """
        new_entries = []
        for policy, Qnet, optimizer in policy_Qnet_optimizers:
            new_policy = deepcopy(policy)
            new_Qnet = deepcopy(Qnet)
            new_optimizer = torch.optim.Adam(new_policy.parameters(), lr=self.lr)
            new_optimizer.load_state_dict(optimizer.state_dict())
            objectives = self.agent.evaluate_policy(self.eval_env, self.seed)
            new_policy.to("cpu")
            new_Qnet.to("cpu")
            for param in new_optimizer.state.values():
                for k, v in param.items():
                    if isinstance(v, torch.Tensor):
                        param[k] = v.cpu()
            new_entries.append((new_policy, new_Qnet, new_optimizer, objectives))
        # 更新非支配集
        for new_entry in new_entries:
            to_remove = []
            add_flag = True
            for idx, entry in enumerate(self.non_dominated_set):
                if is_dominated(new_entry[3], entry[3]):
                    add_flag = False
                    break
                elif is_dominated(entry[3], new_entry[3]):
                    to_remove.append(idx)
            if add_flag:
                # 删除被支配的旧策略
                for idx in reversed(to_remove):
                    del self.non_dominated_set[idx]
                self.non_dominated_set.append(new_entry)
        print(f"SAC Pareto INNER, pareto policies size: {len(self.non_dominated_set)}")

    def HV(self):
        if not self.non_dominated_set:
            return 0.0
        objs = np.array([entry[3] for entry in self.non_dominated_set])
        ref_point = np.zeros((self.num_objectives,))
        HV = hypervolume.InnerHyperVolume(ref_point)
        return HV.compute(objs)

    def SP(self):
        if len(self.non_dominated_set) < 2:
            return 0.0
        objs = np.array([entry[3] for entry in self.non_dominated_set])
        sp = 0.0
        for i in range(self.num_objectives):
            sorted_obj = np.sort(objs[:, i])
            sp += np.sum((sorted_obj[1:] - sorted_obj[:-1]) ** 2)
        return sp / (len(objs) - 1)

    def update_ParetoFront(self):
        for _ in range(self.opposite_num):
            self.updateAEpisode(omega=None, obj_id=self.edge_direction,
                                update_info="++++omega1")  # 不包含目标 obj_id 的 Pareto 上升方向
            self.update_non_dominated_set([(self.agent.policy_net, self.agent.q1_net, self.agent.policy_optimizer)])
        # rewards = self.agent.evaluate_policy(policy)
        # rewards /= self.reward_max
        # omega = ppo.getOmega(rewards)
        for _ in range(self.pareto_ascent_num):
            self.updateAEpisode(omega=None, obj_id=-1, update_info="++++omega2")  # 包含 obj_id 的 Pareto 上升方向
            self.update_non_dominated_set([(self.agent.policy_net, self.agent.q1_net, self.agent.policy_optimizer)])
        self.agent.evaluate_policy(self.eval_env, self.seed)

    def updateAEpisode(self, omega, obj_id=-1, update_info=""):
        state = self.env.reset(seed=self.seed)
        if self.agent.state_normalize:
            state = self.agent.state_norm(state)
        for t in range(self.steps):
            action = self.agent.get_action(state)
            next_state, reward, done, terminated, _ = self.env.step(action)
            if self.agent.state_normalize:
                next_state = self.agent.state_norm(next_state)
            if self.agent.reward_scaling:
                reward = self.agent.reward_scal(reward)
            self.agent.buffer.push((state, action, next_state, reward, terminated))
            state = next_state
            self.agent.update(omega=omega, obj_id=obj_id, update_info=update_info)
            if done:
                state = self.env.reset()
                if self.agent.state_normalize:
                    state = self.agent.state_norm(state)
                if self.agent.reward_scaling:
                    self.agent.reward_scal.reset()

    # ---------------------------> 训练 <-----------------------------
    def train(self):
        with tqdm(total=self.obj_episodes + self.pareto_front_size, desc="Inner Training") as pbar:
            # 寻找内部
            for i in range(self.obj_episodes):
                rewards = self.agent.evaluate_policy(self.eval_env, self.seed)
                rewards = self.reward_max / rewards
                omega = SAC.getOmega(rewards)
                print(f"Inner omega:{omega}")
                self.updateAEpisode(omega=omega, obj_id=-1, update_info="omega:")
                if i > self.obj_episodes / 2:
                    self.update_non_dominated_set(
                        [(self.agent.policy_net, self.agent.q1_net, self.agent.policy_optimizer)])
                pbar.update(1)
            # Pareto front 在线追踪
            new_agent = SAC.SAC(self.state_dim, self.action_dim, self.num_objectives, self.lr)
            SAC.copy_from(self.agent, new_agent)  # 记录备份
            for obj in range(self.num_objectives):
                self.edge_direction = obj
                SAC.copy_from(new_agent, self.agent)  # 恢复备份
                if self.clear_buffer:
                    self.agent.buffer.clear_buffer()  # 清空经验池
                print(f"========== 沿着目标 {self.edge_direction} 的反方向搜索")
                for i in range(int(self.pareto_front_size // self.num_objectives)):
                    self.update_ParetoFront()
                    if i % 10 == 0:
                        current_hv = self.HV()
                        current_sp = self.SP()
                        self.hv_history.append(current_hv)
                        self.sp_history.append(current_sp)
                        print(f"INNER: | HV:{current_hv}, SP:{current_sp}")
                    pbar.update(1)
            self.visualize_training()
            self.visualize_pareto_front()


# 使用示例
if __name__ == "__main__":
    # 初始化时自动检测设备
    env_name = "Walker2d"
    max_episode_steps = 1000
    if env_name == "HalfCheetah":
        env = half_cheetah.HalfCheetah(max_episode_steps=max_episode_steps)
        eval_env = half_cheetah.HalfCheetah(max_episode_steps=max_episode_steps)
        obj_num = 2
    elif env_name == "Hopper-2":
        env = hopper_2.Hopper_2(max_episode_steps=max_episode_steps)
        eval_env = hopper_2.Hopper_2(max_episode_steps=max_episode_steps)
        obj_num = 2
    elif env_name == "Swimmer":
        env = swimmer.Swimmer(max_episode_steps=max_episode_steps)
        eval_env = swimmer.Swimmer(max_episode_steps=max_episode_steps)
        obj_num = 2
    elif env_name == "Ant":
        env = ant.Ant(max_episode_steps=max_episode_steps)
        eval_env = ant.Ant(max_episode_steps=max_episode_steps)
        obj_num = 2
    elif env_name == "Walker2d":
        env = walker2d.Walker2D(max_episode_steps=max_episode_steps)
        eval_env = walker2d.Walker2D(max_episode_steps=max_episode_steps)
        obj_num = 2
    elif env_name == "Humanoid":
        env = humanoid.Humanoid(max_episode_steps=max_episode_steps)
        eval_env = humanoid.Humanoid(max_episode_steps=max_episode_steps)
        obj_num = 2
    else:
        env = hopper_3.Hopper_3(max_episode_steps=max_episode_steps)
        eval_env = hopper_3.Hopper_3(max_episode_steps=max_episode_steps)
        obj_num = 3

    morl = SAC_INNER(env, eval_env, obj_num, lr=3e-4, seed=42)
    print(f"Pareto INNER Training on device: {device}")
    morl.train()
    print("All done!")
