"""配置、图批处理和两个训练阶段共用的短训练循环。"""

import argparse
import copy
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def project_path(path):
    """所有相对路径以项目根目录为基准，编辑器的工作目录不影响结果。"""
    return (ROOT / path).resolve()


def load_config(default_config="configs/case1.json", argv=None):
    parser = argparse.ArgumentParser(description="SE-AGAE 基础流程")
    parser.add_argument("--config", type=Path, default=default_config)
    args = parser.parse_args(argv)
    with project_path(args.config).open(encoding="utf-8") as stream:
        config = json.load(stream)
    for key in ("raw_dir", "graph_dir", "condition_dir", "output_dir", "mesh"):
        config["data"][key] = str(project_path(config["data"][key]))
    return config


def setup(config):
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    name = config["device"]
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name)


def batch_graph(graph, count, device):
    """复制固定网格，使一个 batch 的每一帧都有独立的图。"""
    pos = graph["pos"].to(device)
    edges = graph["edge_index"].to(device)
    n_nodes = len(pos)
    batch = torch.arange(count, device=device).repeat_interleave(n_nodes)
    offsets = torch.arange(count, device=device).view(count, 1, 1) * n_nodes
    edges = (edges.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)
    return pos.repeat(count, 1), edges, batch


def fit(model, train_loader, val_loader, step, config, history_path):
    """step(batch) 返回损失和计数；保存最佳验证轮次，不包含恢复/回退分支。"""
    if config["epochs"] < 1 or config["patience"] < 1:
        raise ValueError("epochs 和 patience 必须为正整数。")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    best, best_state, patience = float("inf"), None, 0
    history = []
    for epoch in range(1, config["epochs"] + 1):
        losses = []
        for training, loader in ((True, train_loader), (False, val_loader)):
            model.train(training)
            total, count = 0.0, 0
            with torch.set_grad_enabled(training):
                for batch in loader:
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                    loss, size = step(batch)
                    if not torch.isfinite(loss):
                        raise ValueError("训练损失出现 NaN/Inf，请检查数据与学习率。")
                    if training:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                        optimizer.step()
                    total += loss.item() * size
                    count += size
            losses.append(total / count)
        history.append({"epoch": epoch, "train_loss": losses[0], "val_loss": losses[1]})
        print(f"Epoch {epoch:04d}: train={losses[0]:.6e}, val={losses[1]:.6e}")
        if losses[1] < best:
            best, patience = losses[1], 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience += 1
        if patience >= config["patience"]:
            break
    model.load_state_dict(best_state)
    model.eval()
    with history_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)


def load_bundle(path):
    # 只加载本项目生成且可信的本地模型文件。
    return torch.load(path, map_location="cpu", weights_only=False)
