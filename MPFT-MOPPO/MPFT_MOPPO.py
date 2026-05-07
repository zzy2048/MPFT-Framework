import multiprocessing as mp
import os
import random
from copy import deepcopy
from multiprocessing import Process, Queue
import numpy as np
import torch
from matplotlib import pyplot as plt
import edge
import hypervolume
import inner
from environments import ant, half_cheetah, hopper_2, hopper_3, humanoid, swimmer, walker2d
from scipy.spatial import Delaunay
from sklearn.decomposition import PCA

# 设置全局设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_dominated(obj1, obj2):
    """ obj1 是否被 obj2 支配 """
    return np.all(obj2 >= obj1) and np.any(obj2 > obj1)


def run_process(target_func, args, q, idx):
    result = target_func(*args)
    q.put((idx, result))


def find_top_k_sparse_regions(arr, topK=2):
    """
    自动选择方法查找 Pareto Front 中的 Top K 稀疏区域：

    - 目标数 = 2：使用排序 + 相邻点距离计算；
    - 目标数 = 3：将三维点投影到最佳拟合平面（通过 PCA 完成）；在二维平面上使用 Delaunay 三角剖分 得到三角网格；
                           计算每个三角形的面积（越大越稀疏）；
    参数:
        arr: shape (n, m)，Pareto Front 的目标值数组，n 是解数量，m 是目标数
        k: 要找的 Top K 个稀疏区域
    返回:
        list of tuples: 每个 tuple 包含 (点1, 点2, 距离)
    """
    n, m = arr.shape
    if m == 2:
        # 二维情况：排序后只考虑相邻点之间的距离
        sorted_arr = arr[arr[:, 0].argsort()]
        distances = np.linalg.norm(sorted_arr[1:] - sorted_arr[:-1], axis=1)

        # 找出前 K 个最大距离的索引
        top_k_indices = np.argsort(distances)[::-1][:topK]

        # 构建结果
        result = []
        for idx in top_k_indices:
            p1 = sorted_arr[idx]
            p2 = sorted_arr[idx + 1]
            result.append([p1, p2])
    else:
        # 3维情况：
        result = find_sparse_regions_3_dim(arr, topK)
    return np.array(result)


def project_to_plane(arr):
    """
    使用 PCA 将三维点投影到最佳拟合的二维平面
    返回：
        arr_proj: 投影后的二维点 (N, 2)
        pca: PCA 模型，可用于还原回三维
    """
    pca = PCA(n_components=2)
    arr_proj = pca.fit_transform(arr)
    return arr_proj, pca


def find_sparse_regions_3_dim(arr, topK=1):
    arr_2d, pca = project_to_plane(arr)
    tri = Delaunay(arr_2d)
    simplices = tri.simplices  # 三角索引 (M, 3)

    # 三角形面积计算
    def area(tri_2d):
        a, b, c = tri_2d
        return 0.5 * np.abs(np.cross(b - a, c - a))

    triangles_2d = arr_2d[simplices]
    areas = np.array([area(tri) for tri in triangles_2d])
    topK_idx = np.argsort(areas)[-topK:]

    # 还原回三维三角形
    sparse_tris_2d = triangles_2d[topK_idx]
    sparse_tris_3d = [pca.inverse_transform(tri) for tri in sparse_tris_2d]
    return np.asarray(sparse_tris_3d)


class mpft_moppo:
    def __init__(self, environment, eval_env, num_objectives, lr=3e-4, seed=42, buffer_size=2048, use_seed=False):
        self.edge1 = None
        self.edge2 = None
        self.edge3 = None
        self.inner = None
        self.topK = 1
        self.pareto_ascent_num = 2
        self.opposite_num = 1
        self.edge_direction = [0, 1, 2]
        self.Warmup_steps = [950, 950]
        self.Xi = [1000, 1000, 1000]
        self.Psi = [1000, 1000, 1000]
        self.J_max = [np.array([1000, 200]), np.array([200, 1000]), np.array([700, 700])]
        self.use_priority_knowledge = True
        self.use_seed = use_seed
        self.seed = seed
        self.lr = lr
        self.steps = buffer_size
        self.env = environment
        self.eval_env = eval_env
        if self.use_seed:
            np.random.seed(self.seed)
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            self.env.action_space.seed(self.seed)  # 动作空间的种子
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.num_objectives = num_objectives
        self.non_dominated_set = []

    # 添加策略保存方法
    def save_policies(self, save_dir="saved_policies"):
        """保存非支配集中的策略到指定目录"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx, (policy, _, objectives) in enumerate(self.non_dominated_set):
            # 生成文件名：策略索引+目标值
            obj_str = "_".join([f"{o:.2f}" for o in objectives])
            filename = f"policy_{idx}_{obj_str}.pth"
            torch.save(policy.state_dict(), os.path.join(save_dir, filename))

    def visualize_pareto_front(self, save_path="pareto_front"):
        """绘制Pareto前沿"""
        save_path += f"_pareto_combine.png"
        if not self.non_dominated_set:
            return
        objectives = np.array([entry[2] for entry in self.non_dominated_set])
        print(f"Objectives: {list(objectives)}")
        plt.figure(figsize=(8, 6))
        if self.num_objectives == 2:
            plt.scatter(objectives[:, 0], objectives[:, 1], s=50, edgecolors='k')
            plt.title(f"2D Pareto Front")
            plt.xlabel("Objective 1")
            plt.ylabel("Objective 2")
        elif self.num_objectives == 3:
            ax = plt.axes(projection='3d')
            ax.scatter3D(objectives[:, 0], objectives[:, 1], objectives[:, 2])
            ax.set_title(f"3D Pareto Front")
            ax.set_xlabel("Objective 1")
            ax.set_ylabel("Objective 2")
            ax.set_zlabel("Objective 3")
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

    def update_non_dominated_set(self, Pareto_set):
        """
        更新非支配解集 self.non_dominated_set 其中元素为：(policy, optimizer, objectives)
        :return:
        """
        new_entries = []
        for policy, optimizer, objectives in Pareto_set:
            new_policy = deepcopy(policy)
            new_optimizer = torch.optim.Adam(new_policy.parameters(), lr=self.lr)
            new_optimizer.load_state_dict(optimizer.state_dict())
            objectives = deepcopy(objectives)
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
            if add_flag:
                # 删除被支配的旧策略
                for idx in reversed(to_remove):
                    del self.non_dominated_set[idx]
                self.non_dominated_set.append(new_entry)
        print(f"Pareto policies size: {len(self.non_dominated_set)}")

    def getJMax(self):
        arr = np.array([obj[2] for obj in self.non_dominated_set])
        n, m = arr.shape
        if n < 3:
            return np.array([m * [3000]] * self.topK)
        result = find_top_k_sparse_regions(arr, self.topK)
        # 按元素取最大
        J_max = np.max(result, axis=1)
        return J_max

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

    def initialize_environment(self, ):
        self.edge1 = edge.EDGE(self.env, self.eval_env, self.num_objectives, lr=self.lr, seed=self.seed,
                               warm_up_steps=1000, Xi=self.Xi[0], Psi=self.Psi[0],
                               edge_direction=self.edge_direction[0], buffer_size=self.steps,
                               pareto_ascent_num=self.pareto_ascent_num,
                               opposite_num=self.opposite_num, J_max=self.J_max[0], use_seed=self.use_seed)
        self.edge2 = edge.EDGE(self.env, self.eval_env, self.num_objectives, lr=self.lr, seed=self.seed,
                               warm_up_steps=1000, Xi=self.Xi[1], Psi=self.Psi[1],
                               edge_direction=self.edge_direction[1], buffer_size=self.steps,
                               pareto_ascent_num=self.pareto_ascent_num,
                               opposite_num=self.opposite_num, J_max=self.J_max[1], use_seed=self.use_seed)
        self.inner = inner.INNER(self.env, self.eval_env, self.num_objectives, lr=self.lr, seed=self.seed,
                                 Xi_k=self.Xi[2],
                                 Psi_k=self.Psi[2], buffer_size=self.steps, pareto_ascent_num=self.pareto_ascent_num,
                                 opposite_num=self.opposite_num, J_max=self.J_max[2], use_seed=self.use_seed)
        if self.num_objectives == 3:
            self.edge3 = edge.EDGE(self.env, self.eval_env, self.num_objectives, lr=self.lr, seed=self.seed,
                                   warm_up_steps=1000, Xi=self.Xi[3], Psi=self.Psi[3],
                                   edge_direction=self.edge_direction[2], buffer_size=self.steps,
                                   pareto_ascent_num=self.pareto_ascent_num,
                                   opposite_num=self.opposite_num, J_max=self.J_max[3], use_seed=self.use_seed)

    @staticmethod
    def _train_edge(edge_class, env, eval_env, num_objectives, lr, seed, warm_up_steps, Xi, Psi, edge_direction, D,
                    pareto_ascent_num, opposite_num, J_max, use_seed):
        """静态方法：并行训练edge"""
        edge = edge_class(env, eval_env, num_objectives, lr=lr, seed=seed, warm_up_steps=warm_up_steps, Xi=Xi,
                          Psi=Psi, edge_direction=edge_direction, buffer_size=D,
                          pareto_ascent_num=pareto_ascent_num,
                          opposite_num=opposite_num, J_max=J_max, use_seed=use_seed)
        edge.train()
        return edge.non_dominated_set

    @staticmethod
    def _train_inner(inner_class, env, eval_env, num_objectives, lr, seed, Xi, Psi, D,
                     pareto_ascent_num, opposite_num, J_max, use_seed):
        """静态方法：并行训练inner"""
        inner = inner_class(env, eval_env, num_objectives, lr=lr, seed=seed, Xi_k=Xi,
                            Psi_k=Psi, buffer_size=D,
                            pareto_ascent_num=pareto_ascent_num,
                            opposite_num=opposite_num, J_max=J_max, use_seed=use_seed)
        inner.train()
        return inner.non_dominated_set

    # ---------------------------> 训练 <-----------------------------
    def train(self):
        task_list = []
        queue = Queue()
        if not self.use_priority_knowledge:
            edge_tasks = [
                (self._train_edge, (edge.EDGE, self.env, self.eval_env, self.num_objectives,
                                    self.lr, self.seed, self.Warmup_steps[0], self.Xi[0], self.Psi[0],
                                    self.edge_direction[0], self.steps,
                                    self.pareto_ascent_num, self.opposite_num, self.J_max[0], self.use_seed)),
                (self._train_edge, (edge.EDGE, self.env, self.eval_env, self.num_objectives,
                                    self.lr, self.seed, self.Warmup_steps[1], self.Xi[1], self.Psi[1],
                                    self.edge_direction[1], self.steps,
                                    self.pareto_ascent_num, self.opposite_num, self.J_max[1], self.use_seed))
            ]
            if self.num_objectives == 3:
                edge_tasks.append(
                    (self._train_edge, (edge.EDGE, self.env, self.eval_env, self.num_objectives,
                                        self.lr, self.seed, self.Warmup_steps[2], self.Xi[3], self.Psi[3],
                                        self.edge_direction[2], self.steps,
                                        self.pareto_ascent_num, self.opposite_num, self.J_max[3], self.use_seed)))
            # 启动 edge 并行训练进程
            for idx, (func, args) in enumerate(edge_tasks):
                p = Process(target=run_process, args=(func, args, queue, idx))
                task_list.append(p)
                p.start()
            # 收集 edge 结果
            for _ in range(len(task_list)):
                idx, result = queue.get()
                print(f"+++++++++第 {idx} 个任务 (edge) 完成+++++++")
                self.update_non_dominated_set(result)

            # 等待所有进程结束
            for p in task_list:
                p.join()

            # inner
            J_max = self.getJMax()
            task_list = []
            queue = Queue()
            inner_tasks = []
            for i in range(self.topK):
                inner_tasks.append((self._train_inner, (inner.INNER, self.env, self.eval_env, self.num_objectives,
                                                        self.lr, self.seed, self.Xi[2], self.Psi[2], self.steps,
                                                        self.pareto_ascent_num,
                                                        self.opposite_num, J_max[i], self.use_seed)))
            # 启动 inner 并行训练进程
            for idx, (func, args) in enumerate(inner_tasks):
                p = Process(target=run_process, args=(func, args, queue, idx))
                task_list.append(p)
                p.start()
            # 收集 inner 结果
            for _ in range(len(task_list)):
                idx, result = queue.get()
                print(f"+++++++++第 {idx} 个任务 (inner) 完成+++++++")
                self.update_non_dominated_set(result)
            # 等待所有进程结束
            for p in task_list:
                p.join()
        else:
            full_tasks = [
                (self._train_edge, (edge.EDGE, self.env, self.eval_env, self.num_objectives,
                                    self.lr, self.seed, self.Warmup_steps[0], self.Xi[0], self.Psi[0],
                                    self.edge_direction[0], self.steps,
                                    self.pareto_ascent_num, self.opposite_num, self.J_max[0], self.use_seed)),
                (self._train_edge, (edge.EDGE, self.env, self.eval_env, self.num_objectives,
                                    self.lr, self.seed, self.Warmup_steps[1], self.Xi[1], self.Psi[1],
                                    self.edge_direction[1], self.steps,
                                    self.pareto_ascent_num, self.opposite_num, self.J_max[1], self.use_seed)),
                (self._train_inner, (inner.INNER, self.env, self.eval_env, self.num_objectives,
                                     self.lr, self.seed, self.Xi[2], self.Psi[2], self.steps, self.pareto_ascent_num,
                                     self.opposite_num, self.J_max[2], self.use_seed))
            ]
            if self.num_objectives == 3:
                full_tasks.append(
                    (self._train_edge, (edge.EDGE, self.env, self.eval_env, self.num_objectives,
                                        self.lr, self.seed, self.Warmup_steps[2], self.Xi[3], self.Psi[3],
                                        self.edge_direction[2], self.steps,
                                        self.pareto_ascent_num, self.opposite_num, self.J_max[3], self.use_seed)))
            for idx, (func, args) in enumerate(full_tasks):
                p = Process(target=run_process, args=(func, args, queue, idx))
                task_list.append(p)
                p.start()
            for _ in range(len(task_list)):
                idx, result = queue.get()
                print(f"+++++++++第 {idx} 个任务 完成+++++++")
                self.update_non_dominated_set(result)
            for p in task_list:
                p.join()

        self.visualize_pareto_front()


# 使用示例
if __name__ == "__main__":
    # 初始化时自动检测设备
    seed = 42  # 42 2 24
    use_seed = False
    steps = 2048
    mp.set_start_method('spawn')
    env_name = "Ant"
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

    morl = mpft_moppo(env, eval_env, obj_num, lr=3e-4, seed=seed, buffer_size=steps, use_seed=use_seed)
    morl.initialize_environment()
    print(f"Training on device: {device}")
    morl.train()

    # 保存训练结果
    # print("\nSaving trained policies...")
    # morl.save_policies()
    # print("Generating visualizations...")
    # morl.visualize_training()
    # morl.visualize_pareto_front()
    print("All done!")

# Hopper-3  steps = 8192
# self.Warmup_steps = [300, 300, 300]
# self.Xi = [500, 500, 500, 500]
# self.Psi = [400, 400, 400, 400]
# self.J_max = [np.array([5500, 4500, 1500]), np.array([4500, 5500, 1500]), np.array([5000, 5000, 2000]),
#                       np.array([4500, 4500, 2500])]

# Swimmer  steps = 2048
# self.Warmup_steps = [0, 300]
# self.Xi = [400, 400, 400]
# self.Psi = [150, 150, 200]
# self.J_max = [np.array([1800, 1600]), np.array([1600, 1800]), np.array([1650, 1650])]


# HalfCheetah  steps = 2048
# self.Warmup_steps = [480, 480]
# self.Xi = [600, 600, 600]
# self.Psi = [800, 800, 800]
# self.J_max = [np.array([1700, 1500]), np.array([1500, 1700]), np.array([1650, 1650])]


# Hopper-2  steps = 8192
# self.Warmup_steps = [180, 180]
# self.Xi = [200, 200, 200]
# self.Psi = [300, 300, 300]
# self.J_max = [np.array([3000., 2500.]), np.array([3000., 2500.]), np.array([2800., 2800.])]


# Ant  steps = 4096
# self.Warmup_steps = [1200, 1200]
# self.Xi = [1200, 1200, 1200]
# self.Psi = [400, 400, 400]
# self.J_max = [np.array([1000, 200]), np.array([200, 1000]), np.array([700, 700])]


# Humanoid  steps = 4096
# self.Warmup_steps = [400, 200]
# self.Xi = [1600, 1000, 1000]
# self.Psi = [500, 900, 900]
# self.J_max = [np.array([4000, 2000]), np.array([2000, 3000]), np.array([3000., 2500.])]


# Walk2D   steps = 2048
# self.Warmup_steps = [500, 300]
# self.Xi = [1500, 600, 800]
# self.Psi = [300, 800, 800]
# self.J_max = [np.array([3000., 4000.]), np.array([3000., 4000.]), np.array([3500., 3250.])]
