"""按单帧物理场训练图自编码器，并按训练编号保存模型与配套数据

输入:
    CONFIG_FILE 或 --config 指定的案例配置
    paths.graph_dir/graph.pt 及 condition_*/<field>.npy，场数组为 [T,N]

输出:
    models/<case>/<field>/ae/<run_id>/autoencoder.pt: 最佳模型及使用所需参数
    runs/<case>/<field>/ae/<run_id>/: history.csv、metrics.csv 和 run.json
    paths.latent_dir/ae/<run_id>/<condition>/z.npy: 潜变量序列 [T,L]
    paths.recon_dir/ae/<run_id>/<condition>/<field>.npy: 按需导出的重构场
    runs 中同阶段的 summary.csv 汇总各次训练；新训练使用独立编号，不覆盖历史结果
    graph、latent、recon 目录相对 paths.data_root，T、N、L 为帧数、节点数和潜变量维度

处理流程:
    按工况划分数据 → 归一化 → 训练并恢复最佳验证模型 → 分别保存模型、潜变量和记录
"""

from functools import lru_cache

import numpy as np
import torch
from torch.utils.data import DataLoader

from core.artifacts import make_run_id, model_path, record_dir, reference, run_session, storage_paths, write_rows
from core.common import batch_graph, fit, load_bundle, load_config, setup
from core.data import FieldFrames, field_stats, load_fields, split_conditions
from core.losses import danger_mask, reconstruction_loss
from core.models import Autoencoder

# 编辑器运行时的默认配置；相对输入路径以项目根目录为基准，--config 可覆盖
CONFIG_FILE = "configs/case_hp_rotor.json"


def train(config):
    """训练自编码器，并将模型、潜变量、重构场和训练记录分类保存"""
    # 1. 初始化运行，读取并校验图与场数据
    # 数据准备前显示编号，便于确认程序已启动
    run_id = make_run_id()
    print(f"Starting AE training | case={config['case']} | field={config['data']['field']} | run={run_id}", flush=True)
    device = setup(config)
    device_text = str(device)
    if device.type == "cuda":
        device_text += f" ({torch.cuda.get_device_name(device)})"
    print(f"Device: {device_text}", flush=True)
    paths = storage_paths(config)
    print("Loading the shared graph and field data...", flush=True)
    graph = load_bundle(paths["graph"] / "graph.pt")
    # 各工况场数组 [T,N] 采用只读内存映射，避免一次加载全部数据
    fields = load_fields(config)
    # 验证场数据的形状与图一致，且配置中指定的重构工况必须存在
    for name, values in fields.items():
        if values.ndim != 2 or values.shape[1] != len(graph["pos"]) or len(values) == 0:
            raise ValueError(f"{name} 场数据应为非空 [T,N]，且 N 与图一致。")
    recon_names = config["export"]["recon_conditions"]
    if set(recon_names) - set(fields):
        raise ValueError("export.recon_conditions 包含不存在的工况。")

    # 2. 划分工况，准备归一化、数据加载器与模型
    # 按完整工况划分；场值统计仅使用训练工况，验证集复用
    split = split_conditions(fields, config["seed"], config["train_ratio"])
    train_frames = sum(len(fields[name]) for name in split["train"])
    val_frames = sum(len(fields[name]) for name in split["val"])
    print(f"Conditions: {len(fields)} (train={len(split['train'])}, val={len(split['val'])}) | "
            f"Frames: train={train_frames}, val={val_frames}", flush=True)
    print("Computing training-field normalization statistics...", flush=True)
    mean, std = field_stats(fields, split["train"])
    print("Normalization statistics ready. Preparing the model...", flush=True)
    # 坐标按各维度独立标准化；torch.std 使用样本标准差，下限避免除零
    pos_mean = graph["pos"].mean(dim=0, keepdim=True)
    pos_std = graph["pos"].std(dim=0, keepdim=True).clamp_min(1e-8)
    graph["pos"] = (graph["pos"] - pos_mean) / pos_std
    ae = config["autoencoder"]
    # FieldFrames 逐帧归一化并可选限幅，batch 为 [B,N,1]，B 为快照数
    loaders = [DataLoader(
        FieldFrames(fields, split[key], mean, std, ae["clamp_normalized"]),
        batch_size=ae["batch_size"],
        shuffle=(key == "train")
        ) 
        for key in ("train", "val")
    ]
    model = Autoencoder(graph["pos"].shape[1], config["model"]).to(device)
    # 危险节点按有限元标签配置，映射下标后扩展 BFS 邻居，生成单图 mask [N]
    mask = danger_mask(
        graph["node_ids"], 
        graph["edge_index"],
        ae["danger_nodes"] if ae["danger_weight"] > 0 else [], 
        ae["danger_neighbors"]
    ).to(device)

    @lru_cache(maxsize=3)
    def graph_for_batch(count):
        """按帧数复用固定批图，最多保留三种大小以限制显存占用"""
        return batch_graph(graph, count, device)

    def step(values):
        """返回归一化场值的重构损失及快照数，用于轮次损失加权"""
        count = len(values)
        # 每帧对应一张独立图，batch 记录展平后各节点所属的快照
        pos, edges, batch = graph_for_batch(count)
        x = values.to(device).reshape(-1, 1)  # [B,N,1] → [B*N,1]
        prediction = model(x, pos, edges, batch)
        return reconstruction_loss(prediction, x, mask, ae["danger_weight"]), count

    # 3. 训练 AE，并恢复最佳验证权重
    with run_session(config, "ae", run_id) as run:
        records = record_dir(config, "ae", run_id)
        latent_dir = paths["latent"] / "ae" / run_id
        latent_dir.mkdir(parents=True, exist_ok=False)
        run["split"] = split
        run["sources"] = {"graph": reference(config, "graph", paths["graph"] / "graph.pt")}
        print(f"Data preparation complete | max_epochs={ae['epochs']} | batch_size={ae['batch_size']} frames", flush=True)
        # fit 返回前切换到评估模式，并保存逐轮记录
        run["training"] = fit(model, *loaders, step, ae, records / "history.csv")

        # 4. 评估最佳模型，导出潜变量和指定工况的重构场
        metrics = []
        val_squares, val_absolute, val_count = 0.0, 0.0, 0
        recon_total, exported = len(set(recon_names)), 0
        print(f"Evaluating the best model and saving latent sequences | conditions={len(fields)} | reconstructions={recon_total}", flush=True)
        with torch.no_grad():
            # 按原帧顺序生成潜变量，保持后续 LSTM 序列对应关系
            for name in fields:
                dataset = FieldFrames(fields, [name], mean, std, ae["clamp_normalized"])
                loader = DataLoader(dataset, batch_size=ae["batch_size"])
                # 完整重构场仅导出配置明确指定的工况，不缓存整个场到内存
                reconstructed = None
                if name in recon_names:
                    folder = paths["recon"] / "ae" / run_id / name
                    folder.mkdir(parents=True, exist_ok=False)
                    reconstructed = np.lib.format.open_memmap(
                        folder / f"{config['data']['field']}.npy", mode="w+",
                        dtype=np.float32, shape=fields[name].shape)
                # frame 是工况内的帧偏移，count 累计场值个数而不是帧数
                parts, squares, absolute, count, frame = [], 0.0, 0.0, 0, 0
                for values in loader:
                    pos, edges, batch = graph_for_batch(len(values))
                    z = model.encoder(values.to(device).reshape(-1, 1), pos, edges, batch)  # [B,L]
                    reconstruction = model.decoder(z, pos, edges, batch).reshape(len(values), -1)
                    # 反归一化后与未限幅的原始场值比较，误差保持字段原始单位
                    physical = reconstruction.cpu().numpy() * std + mean
                    difference = physical.astype(np.float64) - fields[name][frame:frame + len(values)]
                    squares += np.square(difference).sum()
                    absolute += np.abs(difference).sum()
                    count += difference.size
                    if reconstructed is not None:
                        reconstructed[frame:frame + len(values)] = physical
                    frame += len(values)
                    parts.append(z.cpu().numpy())
                # 写回重构场后再报告当前工况导出完成
                if reconstructed is not None:
                    reconstructed.flush()
                    del reconstructed
                    exported += 1
                    print(f"Reconstruction export {exported}/{recon_total}: {name} ({len(fields[name])} frames)", flush=True)
                folder = latent_dir / name
                folder.mkdir()
                np.save(folder / "z.npy", np.concatenate(parts).astype(np.float32))  # [T,L]
                is_train = name in split["train"]
                metrics.append({"condition": name, "split": "train" if is_train else "val",
                                "rmse": float(np.sqrt(squares / count)), "mae": float(absolute / count)})
                if not is_train:
                    val_squares += squares
                    val_absolute += absolute
                    val_count += count

        # 5. 保存最佳模型、评估指标和产物记录
        # 潜变量和完整重构场独立存放，不嵌入模型包
        bundle = {
            # 模型身份与来源
            "case": config["case"], 
            "field": config["data"]["field"], 
            "stage": "ae", 
            "run_id": run_id,
            "model_config": config["model"],
            # 最佳权重转到 CPU 后保存
            "state_dict": {
                key: value.cpu() 
                for key, value in model.state_dict().items()
                },
            # 标准化图和场值归一化参数
            "graph": graph, 
            "field_mean": mean,
            "field_std": std,
        }
        target = model_path(config, "ae", run_id)
        torch.save(bundle, target)
        write_rows(records / "metrics.csv", metrics)
        # 汇总指标按全部验证帧和节点归约，不对工况 RMSE 再做简单平均
        run["metrics"] = {
            "val_rmse": float(np.sqrt(val_squares / val_count)),
            "val_mae": float(val_absolute / val_count)
        }
        # 产物使用逻辑根目录和相对路径引用，便于迁移
        run["artifacts"] = {
            "model": reference(config, "models", target),
            "latent": reference(config, "latent", latent_dir)
        }
        if recon_names:
            run["artifacts"]["recon"] = reference(config, "recon", paths["recon"] / "ae" / run_id)
    # 退出运行上下文后，运行记录和汇总表已写入
    print(f"AE complete | val_rmse={run['metrics']['val_rmse']:.6e} | val_mae={run['metrics']['val_mae']:.6e}\n"
            f"Run logs: {records}", flush=True)
    return run


if __name__ == "__main__":
    train(load_config(CONFIG_FILE))
