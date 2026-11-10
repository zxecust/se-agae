"""损失计算模块：计算重构与潜变量损失，支持危险区域加权和有效帧掩码"""

from collections import deque

import torch
from torch.nn import functional as F


# AE 危险区域选择与重构损失
def danger_mask(node_ids, edge_index, labels, neighbor_count):
    """为每个危险节点扩展最多 neighbor_count 个拓扑邻居，合并为布尔掩码 [N]"""
    # 1. 准备单图掩码、邻接表和标签映射
    # labels 使用原始有限元节点标签，edge_index [2,E] 使用图节点下标
    mask = torch.zeros(len(node_ids), dtype=torch.bool)
    if not labels:
        return mask  # 未配置危险节点时跳过邻接表构建
    # 按无向连接构造邻接表，供广度优先搜索沿拓扑逐层访问
    adjacency = [[] for _ in node_ids]
    for source, target in edge_index.t().tolist():
        adjacency[source].append(target)
        adjacency[target].append(source)
    label_to_index = {int(label): i for i, label in enumerate(node_ids)}

    # 2. 从每个危险节点独立扩展邻居，将结果合并到同一个 mask
    for label in labels:
        start = label_to_index[label]  # 配置中的标签必须属于当前网格
        mask[start] = True
        # visited 防止本次搜索重复访问，布尔 mask 使不同危险区域的重叠节点只计一次
        queue, visited, count = deque([start]), {start}, 0
        # count 不含起点；neighbor_count 限制邻居数量，不表示层数或几何距离
        while queue and count < neighbor_count:
            for neighbor in adjacency[queue.popleft()]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)
                mask[neighbor] = True
                count += 1
                if count == neighbor_count:
                    break
    return mask


def reconstruction_loss(prediction, target, mask, weight):
    """计算全场 MSE，并按权重附加危险区域的独立平均 MSE"""
    # 单通道 prediction、target 为 [B*N,1]，先在全部快照和节点上求均值
    loss = F.mse_loss(prediction.float(), target.float())
    if weight > 0 and mask.any():
        # 单图 mask [N] 按帧重复为 [B*N]，与展平的节点顺序一致
        repeated = mask.repeat(target.numel() // mask.numel())
        # 危险节点仍参与全场项，此处再附加区域项，两个均值分别计算
        loss = loss + weight * F.mse_loss(
            prediction.float().reshape(-1)[repeated], target.float().reshape(-1)[repeated])
    return loss


# LSTM 有效帧损失与逐帧差分约束
def masked_mse(prediction, target, mask):
    """对有效帧的全部潜变量分量求平均平方误差，排除序列补齐位置"""
    # prediction、target 为 [B,T,L]，mask [B,T,1] 广播到各潜变量分量
    # 分母为有效帧数 × L；无有效位置时（如短序列差分）将分母下限设为 1
    return (((prediction - target) ** 2) * mask).sum() / (mask.sum() * prediction.shape[-1]).clamp_min(1)


def latent_loss(prediction, target, mask, mean, std, config):
    """计算 LSTM 潜变量损失，按配置附加一阶、二阶逐帧差分项"""
    # 1. 确定损失尺度并计算基础 MSE，config 为 predictor 子配置
    # 关闭 whiten_loss 时提前返回，差分项也不参与计算
    prediction, target = prediction.float(), target.float()
    if not config["whiten_loss"]:
        return masked_mse(prediction, target, mask)
    # mean、std 来自训练潜变量，形状 [1,1,L]；逐维标准化仅用于损失
    prediction, target = (prediction - mean) / std, (target - mean) / std
    loss = masked_mse(prediction, target, mask)

    # 2. 附加一阶差分损失，约束相邻帧变化量，不除以物理时间间隔
    if config["delta_weight"] > 0:
        # 相邻两帧均有效，差分位置才有效；序列长度由 T 变为 T-1
        loss = loss + config["delta_weight"] * masked_mse(
            prediction[:, 1:] - prediction[:, :-1], target[:, 1:] - target[:, :-1],
            mask[:, 1:] * mask[:, :-1])

    # 3. 附加二阶差分损失，约束相邻变化量的变化，同样按帧计算
    if config["delta2_weight"] > 0:
        # 连续三帧均有效才计入，序列长度由 T 变为 T-2
        loss = loss + config["delta2_weight"] * masked_mse(
            prediction[:, 2:] - 2 * prediction[:, 1:-1] + prediction[:, :-2],
            target[:, 2:] - 2 * target[:, 1:-1] + target[:, :-2],
            mask[:, 2:] * mask[:, 1:-1] * mask[:, :-2])
    return loss
