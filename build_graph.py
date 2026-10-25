"""读取一个案例的 INP 和字段 CSV，构建共享图与逐工况场数组。"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from core.common import load_bundle, load_config
from core.data import finite

# 在编辑器中点击“运行 Python 文件”时使用；命令行 --config 可覆盖。
CONFIG_FILE = "configs/case1.json"


def read_inp(path):
    nodes, elements, mode, pending = {}, [], None, []
    with path.open(encoding="utf-8", errors="ignore") as stream:
        for raw in stream:
            line = raw.strip()
            if not line or line.startswith("**"):
                continue
            if line.startswith("*"):
                if pending:
                    raise ValueError("INP 单元连接行未完成。")
                keyword = line.split(",")[0].lower()
                mode = keyword if keyword in ("*node", "*element") else None
                continue
            parts = [part.strip() for part in line.split(",") if part.strip()]
            if mode == "*node":
                label = int(parts[0])
                if label in nodes:
                    raise ValueError("INP 节点编号重复；请提供单个网格，先合并或重编号多个实例。")
                nodes[label] = [float(value) for value in parts[1:]]
            elif mode == "*element":
                pending.extend(int(value) for value in parts)
                if not line.endswith(","):
                    elements.append(pending[1:])
                    pending = []
    if not nodes or not elements or pending:
        raise ValueError(f"INP 中缺少完整节点或单元：{path}")
    return nodes, elements


def csv_node_ids(columns):
    labels = []
    for column in columns:
        match = re.search(r"\d+", str(column))
        if match is None:
            raise ValueError(f"字段列名无法解析节点编号：{column}")
        labels.append(int(match.group()))
    if len(set(labels)) != len(labels):
        raise ValueError("字段 CSV 含重复节点编号。")
    return labels


def build(config):
    data = config["data"]
    root, output = Path(data["raw_dir"]), Path(data["graph_dir"])
    sources = [folder / f"{data['field']}.csv" for folder in sorted(root.glob("condition_*")) if folder.is_dir()]
    if not sources:
        raise FileNotFoundError(f"没有找到 condition_*/{data['field']}.csv：{root}")
    nodes, elements = read_inp(Path(data["mesh"]))
    reference = pd.read_csv(sources[0], index_col=0)
    node_ids = csv_node_ids(reference.columns)
    graph_path = output / "graph.pt"
    existing = load_bundle(graph_path) if graph_path.is_file() else None
    if existing is not None:
        if set(node_ids) != set(existing["node_ids"].tolist()):
            raise ValueError("当前字段节点与已有图不同，请使用独立 graph_dir。")
        node_ids = existing["node_ids"].tolist()
    index = {label: i for i, label in enumerate(node_ids)}
    # 保留原项目的同单元全连接规则和边生成顺序。
    edges = set()
    for element in elements:
        for i, first in enumerate(element):
            for second in element[i + 1:]:
                edges.add((first, second))
                edges.add((second, first))
    mapped = [(index[a], index[b]) for a, b in edges if a in index and b in index]
    if not mapped:
        raise ValueError("对齐 CSV 节点后，图中没有边。")
    graph = {
        "pos": torch.tensor([nodes[label] for label in node_ids], dtype=torch.float32),
        "edge_index": torch.tensor(mapped, dtype=torch.long).t().contiguous(),
        "node_ids": torch.tensor(node_ids, dtype=torch.long),
    }
    if existing is not None and any(not torch.equal(graph[key], existing[key]) for key in graph):
        raise ValueError("网格与已有图不同，请使用独立 graph_dir。")
    output.mkdir(parents=True, exist_ok=True)
    torch.save(graph, output / "graph.pt")
    np.save(output / "node_ids.npy", node_ids)
    for source in sources:
        table = pd.read_csv(source, index_col=0)
        labels = csv_node_ids(table.columns)
        if set(labels) != set(node_ids):
            raise ValueError(f"字段的节点集合不一致：{source}")
        lookup = {label: i for i, label in enumerate(labels)}
        values = finite(table.iloc[:, [lookup[label] for label in node_ids]].to_numpy(dtype=np.float32), source)
        if len(values) == 0:
            raise ValueError(f"场序列为空：{source}")
        folder = output / source.parent.name
        folder.mkdir(exist_ok=True)
        np.save(folder / f"field_{data['field']}.npy", values)
        # 与原项目一致：数值索引直接保存；Step-Frame 字符串使用配置间隔生成。
        times = pd.to_numeric(table.index, errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(times).all():
            times = np.arange(len(table), dtype=np.float64) * data["dt"]
        np.save(folder / "time.npy", times)
        if source == sources[0]:
            np.save(output / "time.npy", times)
        print(f"{source.parent.name}: field={values.shape}")
    print(f"Graph: nodes={len(node_ids)}, edges={len(mapped)}, saved to {output}")


if __name__ == "__main__":
    build(load_config(CONFIG_FILE))
