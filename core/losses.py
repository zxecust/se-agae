"""全场/危险区域重构损失，以及原项目的潜变量差分损失。"""

from collections import deque

import torch
from torch.nn import functional as F


def danger_mask(node_ids, edge_index, labels, neighbor_count):
    """每个危险节点增加最多 neighbor_count 个 BFS 邻居；重叠节点只计一次。"""
    mask = torch.zeros(len(node_ids), dtype=torch.bool)
    adjacency = [[] for _ in node_ids]
    for source, target in edge_index.t().tolist():
        adjacency[source].append(target)
        adjacency[target].append(source)
    label_to_index = {int(label): i for i, label in enumerate(node_ids)}
    for label in labels:
        start = label_to_index[label]  # 配置中的危险节点必须属于当前网格。
        mask[start] = True
        queue, visited, count = deque([start]), {start}, 0
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
    loss = F.mse_loss(prediction.float(), target.float())
    if weight > 0 and mask.any():
        repeated = mask.repeat(target.numel() // mask.numel())
        loss = loss + weight * F.mse_loss(
            prediction.float().reshape(-1)[repeated], target.float().reshape(-1)[repeated])
    return loss


def masked_mse(prediction, target, mask):
    return (((prediction - target) ** 2) * mask).sum() / (mask.sum() * prediction.shape[-1]).clamp_min(1)


def latent_loss(prediction, target, mask, mean, std, config):
    # 与旧主线一致：关闭 whiten_loss 时只计算原始潜变量 MSE。
    prediction, target = prediction.float(), target.float()
    if not config["whiten_loss"]:
        return masked_mse(prediction, target, mask)
    prediction, target = (prediction - mean) / std, (target - mean) / std
    loss = masked_mse(prediction, target, mask)
    if config["delta_weight"] > 0:
        loss = loss + config["delta_weight"] * masked_mse(
            prediction[:, 1:] - prediction[:, :-1], target[:, 1:] - target[:, :-1],
            mask[:, 1:] * mask[:, :-1])
    if config["delta2_weight"] > 0:
        loss = loss + config["delta2_weight"] * masked_mse(
            prediction[:, 2:] - 2 * prediction[:, 1:-1] + prediction[:, :-2],
            target[:, 2:] - 2 * target[:, 1:-1] + target[:, :-2],
            mask[:, 2:] * mask[:, 1:-1] * mask[:, :-2])
    return loss
