"""数据处理模块：读取物理场与工况特征，组织场快照和变长序列样本"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from core.artifacts import storage_paths


def finite(values, name):
    if not np.isfinite(values).all():
        raise ValueError(f"{name} 含 NaN/Inf。")
    return values


def condition_features(path, config):
    table = pd.read_csv(path)
    values = finite(table[config["columns"]].to_numpy(dtype=np.float32), path)
    if len(values) == 0:
        raise ValueError(f"工况序列为空：{path}")
    if config["difference_columns"]:
        indexes = [config["columns"].index(name) for name in config["difference_columns"]]
        differences = np.zeros((len(values), len(indexes)), dtype=np.float32)
        differences[1:] = values[1:, indexes] - values[:-1, indexes]
        values = np.concatenate([values, differences], axis=1)
    if config["add_frame"]:
        values = np.concatenate([values, np.arange(len(values), dtype=np.float32)[:, None]], axis=1)
    return values


def condition_path(config, name):
    data = config["data"]
    if not data["condition_dir"]:
        raise ValueError("请设置 data.condition_dir，指向工况 CSV 所在目录。")
    return Path(data["condition_dir"]) / data["condition_pattern"].format(condition=name)


def load_fields(config):
    root = storage_paths(config)["graph"]
    paths = sorted(root.glob(f"condition_*/{config['data']['field']}.npy"))
    if not paths:
        raise FileNotFoundError(f"没有场数据，请先运行 build_graph.py：{root}")
    # 按工况名返回只读内存映射，单通道场数组以 [T,N] 存放
    return {path.parent.name: np.load(path, mmap_mode="r") for path in paths}


def split_conditions(names, seed, ratio):
    if len(names) < 2 or not 0 < ratio < 1:
        raise ValueError("至少需要两个工况，train_ratio 必须在 0 与 1 之间。")
    names = sorted(names)
    np.random.RandomState(seed).shuffle(names)
    boundary = max(1, min(round(ratio * len(names)), len(names) - 1))
    return {"train": names[:boundary], "val": names[boundary:]}


def field_stats(fields, names):
    total, squares, count = 0.0, 0.0, 0
    for name in names:
        for start in range(0, len(fields[name]), 16):
            block = finite(np.asarray(fields[name][start:start + 16], dtype=np.float64), name)
            total += block.sum()
            squares += np.square(block).sum()
            count += block.size
    mean = total / count
    return float(mean), float(np.sqrt(max(squares / count - mean ** 2, 0)) + 1e-8)


class FieldFrames(Dataset):
    def __init__(self, fields, names, mean, std, clamp):
        self.fields, self.mean, self.std, self.clamp = fields, mean, std, clamp
        self.index = [(name, frame) for name in names for frame in range(len(fields[name]))]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        name, frame = self.index[index]
        values = finite(np.array(self.fields[name][frame], dtype=np.float32), name)
        values = (values - self.mean) / self.std
        if self.clamp is not None:
            values = np.clip(values, -self.clamp, self.clamp)
        return torch.from_numpy(values[:, None])


class ConditionSequences(Dataset):
    def __init__(self, pairs, mean, std):
        self.pairs, self.mean, self.std = pairs, mean, std

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        condition, latent = self.pairs[index]
        values = (condition - self.mean) / (self.std + 1e-8)
        return torch.from_numpy(values.astype(np.float32)), torch.from_numpy(latent)


def collate_sequences(items):
    lengths = torch.tensor([len(x) for x, _ in items], dtype=torch.long)
    x = torch.nn.utils.rnn.pad_sequence([x for x, _ in items], batch_first=True)
    y = torch.nn.utils.rnn.pad_sequence([y for _, y in items], batch_first=True)
    mask = (torch.arange(x.shape[1])[None, :] < lengths[:, None]).float().unsqueeze(-1)
    return x, y, mask, lengths
