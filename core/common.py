"""通用工具模块：提供配置加载、设备设置、批图构建和公共训练循环"""

import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

# 由当前源码位置确定项目根目录，避免相对路径受启动目录影响
ROOT = Path(__file__).resolve().parents[1]


# 路径与配置处理
def project_path(path):
    """将路径转为绝对路径，相对路径以项目根目录为基准"""
    return (ROOT / path).resolve()


def read_config(path):
    """读取案例配置；网格、原始 CSV 和工况表的位置独立于大型数据根目录"""
    with project_path(path).open(encoding="utf-8") as stream:
        config = json.load(stream)
    # 这里只解析外部输入路径，模型和大型数组的目录由 artifacts.storage_paths 处理
    # raw_dir、condition_dir 和 mesh 分别为原始 CSV、工况表和网格文件的路径，若配置中未指定则保持 None
    for key in ("raw_dir", "condition_dir", "mesh"):
        if config["data"][key]:
            config["data"][key] = str(project_path(config["data"][key]))
    return config


def load_config(default_config="configs/case_hp_rotor.json", argv=None):
    """解析 --config 参数，并读取指定或默认的案例配置"""
    parser = argparse.ArgumentParser(description="SE-AGAE 基础流程")  # 创建命令行解析器
    parser.add_argument("--config", type=Path, default=default_config) # 添加 --config 参数，指定案例配置文件路径
    # argv=None 时读取实际命令行，也可显式传入参数列表供程序内调用
    args = parser.parse_args(argv)
    return read_config(args.config)


# 运行环境设置
def setup(config):
    """设置各随机数生成器的种子，并按配置返回计算设备"""
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # auto 优先使用 CUDA，否则使用 CPU；显式设备名直接交给 torch.device
    name = config["device"]
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name)


# 图批处理
def batch_graph(graph, count, device):
    """为 B=count 帧构造独立批图，返回坐标、边和节点所属帧编号"""
    # 当前项目构造的图数据是 PyTorch 的二进制文件，Python 加载之后是一个字典，包含三个张量，分别是节点坐标、边连接表、节点编号表
    # 获得单图的坐标和边，输入到计算设备
    pos = graph["pos"].to(device) # [N,D]，N 为节点数，D 为坐标维度
    edges = graph["edge_index"].to(device) # [2,E]，E 为边数
    n_nodes = len(pos) # 节点数
    # 节点按帧展开，batch [B*N] 标记每个节点属于哪一帧
    batch = torch.arange(count, device=device).repeat_interleave(n_nodes)
    # 每帧边端点增加帧编号 × N，复制拓扑而不产生跨帧连接
    offsets = torch.arange(count, device=device).view(count, 1, 1) * n_nodes
    # 复制边的关系并调整节点下标
    edges = (edges.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)
    # 坐标 [B*N,D]、边 [2,B*E] 与展平后的场值顺序一致
    return pos.repeat(count, 1), edges, batch


# 公共训练循环
def fit(model, train_loader, val_loader, step, config, history_path):
    """调用 step 训练和验证，恢复最佳权重并返回训练摘要"""
    # 公共的训练流程主要包括：按照轮次读取数据、计算损失、更新参数、记录日志、保存最佳权重和训练历史
    # 输入的参数未要训练的模型、逐批提供的训练数据、逐批提供的验证数据、训练和验证的 step 函数、训练配置和训练历史保存路径
    
    # 1. 校验轮次配置，初始化优化器和训练记录
    if config["epochs"] < 1 or config["patience"] < 1:
        # 训练轮次数和早停耐心值必须为正整数，否则无法进行训练
        raise ValueError("epochs 和 patience 必须为正整数。")
    # 初始化 Adam 优化器，使用配置中的学习率和权重衰减参数
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    # 以下四个变量依次为最佳验证损失、最佳权重状态、连续未改善轮数和最佳轮次
    best, best_state, patience, best_epoch = float("inf"), None, 0, 0
    history = []
    device = next(model.parameters()).device

    # 2. 逐轮训练与验证，记录耗时并选择最佳权重
    # 外层循环控制训练轮次，除非满足早停的条件，否则会继续训练直到达到配置的最大轮次
    for epoch in range(1, config["epochs"] + 1):
        # CUDA 同步使计时覆盖本轮训练与验证，不计入上一阶段尚未完成的工作
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        losses = [] # 记录训练和验证的损失值，分别对应 train_loss 和 val_loss
        # 先训练再验证；模型模式与梯度开关分别控制，验证阶段不更新参数
        for training, loader in ((True, train_loader), (False, val_loader)):
            # 切换模型模式，训练模式启用 dropout 和 batchnorm，验证模式禁用
            model.train(training) 
            total, count = 0.0, 0
            # torch.set_grad_enabled 根据 training 决定是否计算梯度，训练阶段需要梯度以便反向传播，验证阶段不需要梯度以节省内存和计算
            with torch.set_grad_enabled(training):
                # 内层循环逐个处理 batch
                for batch in loader:
                    if training:
                        optimizer.zero_grad(set_to_none=True)  # 每批清除上一批梯度
                    # step 负责前向计算与损失，size 决定轮次日志的统计权重
                    loss, size = step(batch)
                    if not torch.isfinite(loss):
                        raise ValueError("训练损失出现 NaN/Inf，请检查数据与学习率。")
                    if training:
                        # 反向传播后裁剪梯度，再由 Adam 更新参数
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                        optimizer.step()
                    # AE 返回快照数，LSTM 返回 1，分别对应按快照或按 batch 加权
                    total += loss.item() * size
                    count += size
            losses.append(total / count)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        # 计算本轮训练和验证的耗时，并记录到历史中
        epoch_seconds = time.perf_counter() - started
        history.append({"epoch": epoch, "train_loss": losses[0], "val_loss": losses[1],
                        "epoch_seconds": epoch_seconds})
        print(f"Epoch {epoch:04d}: "
                f"train={losses[0]:.6e}, val={losses[1]:.6e}, time={epoch_seconds:.2f}s", flush=True)
        # 根据验证损失判断是否更新最佳权重和早停计数
        if losses[1] < best:
            best, patience = losses[1], 0
            best_epoch = epoch
            # 独立复制最佳状态到内存，避免后续参数更新改写这份快照
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience += 1
        # patience 是连续未改善轮数，达到配置上限后停止
        if patience >= config["patience"]:
            print(f"Early stopping: validation loss has not improved for {patience} epochs.", flush=True)
            break

    # 3. 恢复最佳权重，保存逐轮历史并返回摘要
    # 此处只恢复内存中的模型，模型文件由入口脚本保存
    model.load_state_dict(best_state)
    model.eval() # 设置为评估模式，禁用 dropout 和 batchnorm
    with history_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "train_loss", "val_loss", "epoch_seconds"])
        writer.writeheader()
        writer.writerows(history)
    print(f"Training finished | epochs={len(history)} | best_epoch={best_epoch} | best_val_loss={best:.6e}", flush=True)
    return {"best_epoch": best_epoch, "best_val_loss": best, "epochs": len(history)}


# 图与模型包加载
def load_bundle(path):
    """将本项目保存的图或模型包加载到 CPU"""
    # 只加载本项目生成且可信的本地模型文件
    return torch.load(path, map_location="cpu", weights_only=False)
