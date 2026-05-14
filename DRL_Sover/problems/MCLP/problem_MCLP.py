from PIL.PngImagePlugin import is_cid
from torch.utils.data import Dataset
import torch
import os
import pickle
import math
import  numpy as np
from problems.MCLP.state_MCLP import StateMCLP



class MCLP(object):
    NAME = 'MCLP'
    @staticmethod
    def get_total_num(dataset, pi):
        w1 = 0.6
        w2 = 0.4
        da = 1000 / 38686.093359901104
        db = 2000 / 38686.093359901104
        users = dataset['users']
        facilities = dataset['facilities']
        demand = dataset['demand']  # 假设形状为 [batch_size, n_user, 1]
        radius = dataset['r'][0]
        batch_size, n_user, _ = users.size()
        _, n_facilities, _ = facilities.size()
        _, p = pi.size()

        # 计算距离
        # dist = (facilities[:, :, None, :] - users[:, None, :, :]).norm(p=2, dim=-1)
        dist = (facilities[:, :, None, :2] - users[:, None, :, :]).norm(p=2, dim=-1)

        # 获取选中的设施与用户之间的距离
        facility_tensor = pi.unsqueeze(-1).expand(-1, -1, n_user)
        f_u_dist_tensor = dist.gather(1, facility_tensor)

        # 计算覆盖情况
        mask = f_u_dist_tensor < radius

        def F(dij, da, db):
            result = torch.zeros_like(dij)
            mask1 = dij <= da
            mask2 = (dij > da) & (dij <= db)
            mask3 = dij > db

            result[mask1] = 1.0
            if (db - da) > 1e-8:  # 避免除以零
                angle = (math.pi / (db - da)) * (dij[mask2] - (da + db) / 2) + math.pi / 2
                result[mask2] = 0.5 * torch.cos(angle) + 0.5
            result[mask3] = 0.0
            return result

        F_dij = F(f_u_dist_tensor, da, db)


        w_j = demand.squeeze(-1)

        # coverage_sum = (F_dij * mask).max(dim=1)[0]
        # print(f"coverage_sum: {coverage_sum},coverage_sum.min: {coverage_sum.min()}, coverage_sum.max: {coverage_sum.max()}")
        #
        # term1_sum = (w_j * (F_dij * mask).max(dim=1)[0]).sum(dim=1)  # [batch_size]
        #
        # term2 = (f_u_dist_tensor * mask.float()).sum(dim=(1, 2))  # [batch_size]
        #
        # # 5. 组合目标函数
        # weighted_covernum = w1 * term1_sum - w2 * term2
        #
        # return weighted_covernum

        # 1. 计算每个用户的最大收益值，以及提供该值的设施索引
        F_masked = F_dij * mask.float()  # [batch, p, n_user]
        best_F_values, best_facility_indices = F_masked.max(dim=1)  # [batch, n_user]

        # 2. 计算收益项 (您的 term1_sum 逻辑已经等价于此，可以保持)
        term1_sum = (w_j * best_F_values).sum(dim=1)  # [batch_size]

        # 3. 根据找到的最佳设施索引，提取对应的距离
        # f_u_dist_tensor 的形状是 [batch, p, n_user]
        # 我们需要从 p 这个维度上，根据 best_facility_indices 来选择距离
        best_distances = f_u_dist_tensor.gather(1, best_facility_indices.unsqueeze(1)).squeeze(1)  # [batch, n_user]

        # 4. 对于未被覆盖的用户，其距离成本应为0
        # best_F_values > 0 可以作为用户是否被覆盖的判断依据
        covered_mask = (best_F_values > 0).float()
        term2_sum = (best_distances * covered_mask).sum(dim=1)  # [batch_size]

        # 5. 组合目标函数
        weighted_covernum = w1 * term1_sum - w2 * term2_sum

        return weighted_covernum

    # @staticmethod
    # def get_total_num(dataset, pi):
    #     """
    #     计算设施选址问题的目标函数值（归一化版本）
    #     目标：最大化 (0.5/W)×Σw_i·y_i - (0.5/C)×Σc_j·x_j
    #
    #     参数:
    #     - dataset: 包含所有必要数据的字典
    #     - pi: 选中的设施索引 [batch_size, p]
    #
    #     返回:
    #     - objective: 归一化的目标函数值 [batch_size]
    #     """
    #     # 提取数据
    #     costs = dataset['costs']  # c_j: 设施j的租金成本 [batch_size, n_facilities]
    #     users = dataset['users']  # 用户位置 [batch_size, n_user, 2]
    #     facilities = dataset['facilities']  # 设施位置 [batch_size, n_facilities, 2]
    #     demand = dataset['demand']  # w_i: 用户i的需求/权重 [batch_size, n_user, 1]
    #     radius = dataset['r'][0]  # R_j: 设施服务半径
    #
    #
    #
    #     # 获取维度信息
    #     batch_size, n_user, _ = users.size()
    #     _, n_facilities, _ = facilities.size()
    #     _, p = pi.size()  # p是选中的设施数量
    #
    #     # 确保demand的维度正确
    #     if demand.dim() == 2:
    #         demand = demand.unsqueeze(-1)  # 确保是 [batch_size, n_user, 1]
    #
    #     # 1. 计算所有设施到所有用户的距离 d_ij
    #     dist_all = (facilities[:, :, None, :2] - users[:, None, :, :]).norm(p=2, dim=-1)
    #     # dist_all: [batch, n_facilities, n_user]
    #
    #     # 2. 获取选中设施的距离
    #     # pi: [batch_size, p], 每个值是0到n_facilities-1的索引
    #     # 扩展pi以便gather操作
    #     pi_expanded = pi.unsqueeze(-1).expand(-1, -1, n_user)  # [batch, p, n_user]
    #     selected_dist = dist_all.gather(1, pi_expanded)
    #     # selected_dist: [batch, p, n_user]
    #
    #     # 3. 判断用户是否被覆盖 (y_i)
    #     # 如果任何选中的设施j满足 d_ij <= R_j，则用户i被覆盖
    #     coverage_mask = selected_dist <= radius  # [batch, p, n_user]
    #     y_i = coverage_mask.any(dim=1).float()  # [batch, n_user]，1表示被覆盖，0表示未被覆盖
    #
    #     # 4. 计算覆盖收益项 Σw_i·y_i
    #     w_i = demand.squeeze(-1)  # [batch, n_user]
    #     coverage_sum = (w_i * y_i).sum(dim=1)  # [batch]
    #
    #     # 6. 计算选中设施的总租金成本 Σc_j·x_j
    #     # 确保costs的维度正确
    #     if costs.dim() == 1:
    #         costs = costs.unsqueeze(0)  # 添加batch维度 [1, n_facilities]
    #         costs = costs.expand(batch_size, -1)  # [batch_size, n_facilities]
    #
    #     selected_costs = costs.gather(1, pi)  # [batch, p]
    #     cost_sum = selected_costs.sum(dim=1)  # [batch]
    #
    #
    #     # 8. 计算归一化的目标函数值
    #     # Maximize: (0.5/W) × Σw_i·y_i - (0.5/C) × Σc_j·x_j
    #     normalized_coverage = 0.7 * coverage_sum
    #     normalized_cost = 0.3 * cost_sum
    #
    #     objective = normalized_coverage - normalized_cost
    #
    #     return objective
    # @staticmethod
    # def get_total_num(dataset, pi):
    #     """
    #     计算给定设施选址方案 (pi) 的 BCLP (Backup Coverage Location Problem) 目标函数值。
    #     该实现与提供的 Gurobi 模型逻辑一致。
    #
    #     Args:
    #         dataset (dict): 包含 'users', 'facilities', 'demand', 'r' 的字典。
    #         pi (torch.Tensor): 选定的设施索引，形状为 [batch_size, p]。
    #
    #     Returns:
    #         torch.Tensor: 每个批次样本的目标函数值，形状为 [batch_size]。
    #     """
    #     # 1. 设置参数
    #     # 根据 Gurobi 代码: m.setObjective(0.8 * term1 + 0.2 * term2, ...)
    #     w = 0.8
    #
    #     # 从数据集中解包所需张量
    #     users = dataset['users']  # 用户/需求点位置 [batch, n_user, 2]
    #     facilities = dataset['facilities']  # 候选设施位置 [batch, n_facilities, 2]
    #     demand = dataset['demand']  # 用户需求量 [batch, n_user, 1]
    #     radius = dataset['r'][0]  # 覆盖半径 (标量)
    #
    #     # 获取维度信息
    #     batch_size, n_user, _ = users.size()
    #     _, n_facilities, _ = facilities.size()
    #     _, p = pi.size()  # p 是选定的设施数量 (K in Gurobi)
    #
    #     # 2. 计算所有候选设施与所有用户之间的距离矩阵
    #     # dist 形状: [batch_size, n_facilities, n_user]
    #     dist = (facilities[:, :, None, :] - users[:, None, :, :]).norm(p=2, dim=-1)
    #
    #     # 3. 根据半径确定二元覆盖矩阵 (N_i 在公式中)
    #     # 如果 facility j 覆盖 user i，则 coverage_matrix[i, j] = 1，否则为 0
    #     # coverage_matrix 形状: [batch_size, n_facilities, n_user]
    #     coverage_matrix = (dist <= radius).float()
    #
    #     # 4. 根据给定的选址方案 (pi)，计算每个用户被覆盖的次数
    #     # pi 形状: [batch_size, p]
    #     # 我们需要从 coverage_matrix 中收集与选定设施对应的覆盖信息
    #
    #     # 将 pi 扩展以便用于 gather
    #     # pi_expanded 形状: [batch_size, p, n_user]
    #     pi_expanded = pi.unsqueeze(-1).expand(-1, -1, n_user)
    #
    #     # selected_coverage 形状: [batch_size, p, n_user]
    #     # 它表示每个用户是否被 p 个选定设施中的每一个所覆盖
    #     selected_coverage = coverage_matrix.gather(1, pi_expanded)
    #
    #     # 对每个用户，计算覆盖它的已选设施总数 (对应公式中的 sum_{j in N_i} x_j)
    #     # coverage_count 形状: [batch_size, n_user]
    #     coverage_count = selected_coverage.sum(dim=1)
    #
    #     # 5. 根据 Gurobi 约束和变量定义，计算 y_i 和 u_i
    #     # y_i = 1 如果用户 i 被至少一个设施覆盖 (coverage_count >= 1)
    #     # u_i = 1 如果用户 i 被至少两个设施覆盖 (coverage_count >= 2)
    #
    #     # y 形状: [batch_size, n_user]
    #     y = (coverage_count >= 1).float()
    #
    #     # u 形状: [batch_size, n_user]
    #     u = (coverage_count >= 2).float()
    #
    #     # 6. 计算目标函数
    #     # 目标函数: Maximize Z = w * sum(a_i * y_i) + (1 - w) * sum(a_i * u_i)
    #
    #     # demand (a_i) 需要从 [batch, n_user, 1] 压缩到 [batch, n_user]
    #     w_j = demand.squeeze(-1)
    #
    #     # 计算第一项
    #     term1 = (w_j * y).sum(dim=1)  # 形状: [batch_size]
    #
    #     # 计算第二项
    #     term2 = (w_j * u).sum(dim=1)  # 形状: [batch_size]
    #
    #     # 组合成最终的目标值
    #     objective_value = w * term1 + (1 - w) * term2
    #
    #     return objective_value


    @staticmethod
    def make_dataset(*args, **kwargs):
        return MCLPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateMCLP.initialize(*args, **kwargs)


class MCLPDataset(Dataset):
    def __init__(self, filename=None, n_users=50, n_facilities=20, num_samples=5000, offset=0, p=8, r=0.2, distribution=None):
        super(MCLPDataset, self).__init__()

        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [row for row in (data[offset:offset + num_samples])]
                p = self.data[0]['p']
                r = self.data[0]['r']
        else:
            # Sample points randomly in [0, 1] square
            self.data = [dict(users=torch.FloatTensor(n_users, 2).uniform_(0, 1),
                              facilities=torch.FloatTensor(n_facilities, 2).uniform_(0, 1),
                              demand=torch.FloatTensor(n_users, 1).uniform_(1, 10),
                              p=p,
                              r=r)
                         for i in range(num_samples)]

        self.size = len(self.data)
        self.p = p
        self.r = r

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]