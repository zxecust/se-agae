"""阶段一：按场快照训练自编码器，导出配套的潜变量序列。"""

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from core.common import batch_graph, fit, load_bundle, load_config, setup
from core.data import FieldFrames, field_stats, load_fields, split_conditions
from core.losses import danger_mask, reconstruction_loss
from core.models import Autoencoder

# 在编辑器中点击“运行 Python 文件”时使用；命令行 --config 可覆盖。
CONFIG_FILE = "configs/case1.json"


def train(config):
    device = setup(config)
    output = Path(config["data"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    graph = load_bundle(Path(config["data"]["graph_dir"]) / "graph.pt")
    fields = load_fields(config)
    for name, values in fields.items():
        if values.ndim != 2 or values.shape[1] != len(graph["pos"]) or len(values) == 0:
            raise ValueError(f"{name} 场数据应为非空 [T,N]，且 N 与图一致。")
    split = split_conditions(fields, config["seed"], config["train_ratio"])
    mean, std = field_stats(fields, split["train"])
    # 坐标标准化沿用旧项目 torch.std 的样本标准差约定。
    pos_mean = graph["pos"].mean(dim=0, keepdim=True)
    pos_std = graph["pos"].std(dim=0, keepdim=True).clamp_min(1e-8)
    graph["pos"] = (graph["pos"] - pos_mean) / pos_std
    ae = config["autoencoder"]
    loaders = [DataLoader(FieldFrames(fields, split[key], mean, std, ae["clamp_normalized"]),
                          batch_size=ae["batch_size"], shuffle=(key == "train")) for key in ("train", "val")]
    model = Autoencoder(graph["pos"].shape[1], config["model"]).to(device)
    mask = danger_mask(graph["node_ids"], graph["edge_index"],
                       ae["danger_nodes"] if ae["danger_weight"] > 0 else [], ae["danger_neighbors"]).to(device)

    def step(values):
        count = len(values)
        pos, edges, batch = batch_graph(graph, count, device)
        x = values.to(device).reshape(-1, 1)
        prediction = model(x, pos, edges, batch)
        return reconstruction_loss(prediction, x, mask, ae["danger_weight"]), count

    print(f"AE: device={device}, train={split['train']}, val={split['val']}")
    fit(model, *loaders, step, ae, output / "ae_history.csv")
    latents, metrics = {}, []
    with torch.no_grad():
        for name in fields:
            dataset = FieldFrames(fields, [name], mean, std, ae["clamp_normalized"])
            loader = DataLoader(dataset, batch_size=ae["batch_size"])
            parts, squares, absolute, count, frame = [], 0.0, 0.0, 0, 0
            for values in loader:
                pos, edges, batch = batch_graph(graph, len(values), device)
                z = model.encoder(values.to(device).reshape(-1, 1), pos, edges, batch)
                reconstruction = model.decoder(z, pos, edges, batch).reshape(len(values), -1)
                physical = reconstruction.cpu().numpy() * std + mean
                difference = physical.astype(np.float64) - fields[name][frame:frame + len(values)]
                squares += np.square(difference).sum()
                absolute += np.abs(difference).sum()
                count += difference.size
                frame += len(values)
                parts.append(z.cpu().numpy())
            latents[name] = np.concatenate(parts).astype(np.float32)
            folder = output / "embedding" / name
            folder.mkdir(parents=True, exist_ok=True)
            np.save(folder / "z.npy", latents[name])
            metrics.append({"condition": name, "split": "train" if name in split["train"] else "val",
                            "rmse": float(np.sqrt(squares / count)), "mae": float(absolute / count)})
    # 把潜变量与生成它的 AE 放在同一个文件中，LSTM 阶段不会混用旧缓存。
    bundle = {
        "config": config, "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
        "graph": graph, "position_mean": pos_mean, "position_std": pos_std,
        "field_mean": mean, "field_std": std, "split": split, "latents": latents,
    }
    torch.save(bundle, output / "autoencoder.pt")
    with (output / "split.json").open("w", encoding="utf-8") as stream:
        json.dump(split, stream, indent=2)
    with (output / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2, ensure_ascii=False)
    with (output / "ae_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)
    print(f"AE 和潜变量已保存：{output}")


if __name__ == "__main__":
    train(load_config(CONFIG_FILE))
