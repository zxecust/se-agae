"""数据处理模块：读取物理场与工况特征，组织场快照和变长序列样本"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from core.artifacts import storage_paths


# 数据检查与工况特征
def finite(values, name):
    """检查数值是否全部有限，通过后原样返回，name 用于定位错误来源"""
    if not np.isfinite(values).all():
        raise ValueError(f"{name} 含 NaN/Inf。")
    return values


def condition_features(path, config):
    """按特征配置提取工况序列 [T,F]，可追加逐帧差分和帧编号"""
    # config 为 features 子配置，columns 的顺序决定基础特征列顺序
    table = pd.read_csv(path) # 读取工况 CSV，列顺序不保证与配置一致
    values = finite(table[config["columns"]].to_numpy(dtype=np.float32), path)
    # 检查工况序列是否为空，避免后续差分或拼接报错
    if len(values) == 0:
        raise ValueError(f"工况序列为空：{path}")
    # 差分列追加到基础特征之后，形状为 [T,K]，K 为所选差分特征数
    # 确定哪些基础特征需要计算差分
    if config["difference_columns"]:
        indexes = [config["columns"].index(name) for name in config["difference_columns"]]
        differences = np.zeros((len(values), len(indexes)), dtype=np.float32)
        # 首帧无前一帧，差分置零；其余为相邻行差值，不除以物理时间间隔
        differences[1:] = values[1:, indexes] - values[:-1, indexes]
        values = np.concatenate([values, differences], axis=1)
    # 最后一列可附加 0..T-1 的帧编号，本函数不归一化或重采样
    # 根据配置决定是否添加帧编号，便于后续按时间顺序处理或绘图
    if config["add_frame"]:
        values = np.concatenate([values, np.arange(len(values), dtype=np.float32)[:, None]], axis=1)
    return values


def condition_path(config, name):
    """按工况名填充文件名模板，生成工况 CSV 路径"""
    data = config["data"]
    # 检查工况目录是否为空
    if not data["condition_dir"]:
        raise ValueError("请设置 data.condition_dir，指向工况 CSV 所在目录。")
    return Path(data["condition_dir"]) / data["condition_pattern"].format(condition=name)


# 场数据读取、划分与统计
def load_fields(config):
    """按工况名返回场数组映射，读取 build_graph 生成的所选字段"""
    # 函数主要读取案例名、物理场名、数据目录等配置
    root = storage_paths(config)["graph"]
    paths = sorted(root.glob(f"condition_*/{config['data']['field']}.npy"))
    if not paths:
        raise FileNotFoundError(f"没有场数据，请先运行 build_graph.py：{root}")
    # [T,N] 中 T 为帧数、N 为节点数；只读内存映射避免一次加载全部场值
    # 列顺序由 build_graph 对齐，此处不再重排节点
    return {path.parent.name: np.load(path, mmap_mode="r") for path in paths} # 按照只读内存映射方式加载场数据；返回的是工况和场数组组成的字典


def split_conditions(names, seed, ratio):
    """按完整工况划分训练集和验证集，返回两组工况名列表"""
    if len(names) < 2 or not 0 < ratio < 1:
        raise ValueError("至少需要两个工况，train_ratio 必须在 0 与 1 之间。")
    # names 可传名称列表或以工况名为键的字典，排序后得到独立列表，不修改原输入
    names = sorted(names)
    # 先排序再按固定种子打乱，使划分不依赖输入顺序，也不影响全局随机状态
    np.random.RandomState(seed).shuffle(names)
    # ratio 为训练工况占比，取整后将训练工况数限制在 [1,工况总数-1]
    boundary = max(1, min(round(ratio * len(names)), len(names) - 1))
    # 前 boundary 个工况用于训练，其余用于验证；仅返回名称，不复制或移动场数据
    return {"train": names[:boundary], "val": names[boundary:]}


def field_stats(fields, names):
    """统计指定工况全部帧和节点的全局均值与总体标准差"""
    # names 由调用方指定，训练时只传入训练工况，验证集复用统计结果
    total, squares, count = 0.0, 0.0, 0 # 场值之和、平方和、总个数（总帧数 × 节点数）
    # 逐工况、分块读取处理数据
    for name in names: # 外部循环按照工况逐个处理
        # 每次处理最多 16 帧，以 float64 累计，避免复制完整场序列
        for start in range(0, len(fields[name]), 16):
            block = finite(np.asarray(fields[name][start:start + 16], dtype=np.float64), name)
            total += block.sum()
            squares += np.square(block).sum()
            count += block.size  # 统计场值个数，即帧数 × 节点数
    mean = total / count
    # 用二阶矩计算总体方差，截去舍入负值，并给标准差加小量避免除零
    return float(mean), float(np.sqrt(max(squares / count - mean ** 2, 0)) + 1e-8)


# AE 单帧样本与 LSTM 序列样本
class FieldFrames(Dataset):
    """AE 数据集，一个样本是一帧归一化物理场，返回张量 [N,1]"""

    def __init__(self, fields, names, mean, std, clamp):
        self.fields, self.mean, self.std, self.clamp = fields, mean, std, clamp
        # 样本索引只保存工况名和工况内帧编号，场值在取样时按需读取
        self.index = [(name, frame) for name in names for frame in range(len(fields[name]))]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        # index 是跨工况的样本下标，frame 是工况内帧编号，均不是节点标签
        name, frame = self.index[index]
        # 复制只读场值为独立 float32 数组，后续处理不修改原始文件
        values = finite(np.array(self.fields[name][frame], dtype=np.float32), name)
        # 使用训练场值的全局统计量归一化，可选限幅只作用于当前样本
        values = (values - self.mean) / self.std
        if self.clamp is not None:
            values = np.clip(values, -self.clamp, self.clamp)
        return torch.from_numpy(values[:, None])  # [N] → [N,1]，增加单通道维度


class ConditionSequences(Dataset):
    """LSTM 数据集，一个样本为完整工况的特征 [T,F] 和目标潜变量 [T,L]"""

    def __init__(self, pairs, mean, std):
        self.pairs, self.mean, self.std = pairs, mean, std

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        condition, latent = self.pairs[index]
        # 特征逐列归一化，mean、std 为训练工况统计量，潜变量保持 AE 原始输出
        values = (condition - self.mean) / (self.std + 1e-8)
        return torch.from_numpy(values.astype(np.float32)), torch.from_numpy(latent)


def collate_sequences(items):
    """合并变长工况样本，返回补齐特征、目标、有效帧掩码和实际长度"""
    # B 为本批工况数，实际长度 [B] 保留在 CPU，供 LSTM 序列打包使用
    lengths = torch.tensor([len(x) for x, _ in items], dtype=torch.long)
    # 右侧补零到本批最长帧数 Tmax，保留各工况原有帧顺序
    # 特征 x [B,Tmax,F]、目标 y [B,Tmax,L]，批维在最前
    x = torch.nn.utils.rnn.pad_sequence([x for x, _ in items], batch_first=True)
    y = torch.nn.utils.rnn.pad_sequence([y for _, y in items], batch_first=True)
    # 帧编号与各工况长度广播比较，有效帧为 1、补齐帧为 0
    # mask [B,Tmax,1] 可广播到所有潜变量分量，损失据此排除补齐位置
    mask = (torch.arange(x.shape[1])[None, :] < lengths[:, None]).float().unsqueeze(-1)
    return x, y, mask, lengths
