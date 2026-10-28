"""配置、图批处理和两个训练阶段共用的短训练循环"""

import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def project_path(path):
    """将路径解析到项目根目录，不依赖编辑器的工作目录"""
    return (ROOT / path).resolve()


def read_config(path):
    """读取案例配置；网格、原始 CSV 和工况表的位置独立于大型数据根目录"""
    with project_path(path).open(encoding="utf-8") as stream:
        config = json.load(stream)
    for key in ("raw_dir", "condition_dir", "mesh"):
        if config["data"][key]:
            config["data"][key] = str(project_path(config["data"][key]))
    return config


def load_config(default_config="configs/case_hp_rotor.json", argv=None):
    parser = argparse.ArgumentParser(description="SE-AGAE 基础流程")
    parser.add_argument("--config", type=Path, default=default_config)
    args = parser.parse_args(argv)
    return read_config(args.config)


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
    """按 step 的计数统计损失与单轮耗时，恢复最佳验证权重并返回最佳轮次信息"""
    if config["epochs"] < 1 or config["patience"] < 1:
        raise ValueError("epochs 和 patience 必须为正整数。")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    best, best_state, patience, best_epoch = float("inf"), None, 0, 0
    history = []
    device = next(model.parameters()).device
    for epoch in range(1, config["epochs"] + 1):
        # CUDA 同步使计时覆盖本轮训练与验证，不计入上一阶段尚未完成的工作
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
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
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_seconds = time.perf_counter() - started
        history.append({"epoch": epoch, "train_loss": losses[0], "val_loss": losses[1],
                        "epoch_seconds": epoch_seconds})
        print(f"[TRAIN][Epoch {epoch:04d}/{config['epochs']:04d}] "
              f"train={losses[0]:.6e}, val={losses[1]:.6e}, time={epoch_seconds:.2f}s", flush=True)
        if losses[1] < best:
            best, patience = losses[1], 0
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience += 1
        if patience >= config["patience"]:
            print(f"[STOP] 验证损失连续 {patience} 轮未改善，提前停止", flush=True)
            break
    model.load_state_dict(best_state)
    model.eval()
    with history_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "train_loss", "val_loss", "epoch_seconds"])
        writer.writeheader()
        writer.writerows(history)
    print(f"[BEST] 训练结束 | 实际轮数={len(history)} | 最佳轮次={best_epoch} | val={best:.6e}", flush=True)
    return {"best_epoch": best_epoch, "best_val_loss": best, "epochs": len(history)}


def load_bundle(path):
    # 只加载本项目生成且可信的本地模型文件
    return torch.load(path, map_location="cpu", weights_only=False)
