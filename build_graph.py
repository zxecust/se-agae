"""将单网格、单字段的逐工况物理场转换为共享图和 NumPy 数组。

输入:
    data.mesh: 含节点坐标和单元连接的 Abaqus INP 文件。
    data.raw_dir/condition_*/<field>.csv: 第一列为帧标识，其余列为节点场值。

输出（位于 data.graph_dir，已有同名文件会被覆盖）:
    graph.pt: 包含 pos [N,D]、edge_index [2,E]、node_ids [N] 的字典。
    node_ids.npy: 统一顺序的有限元节点标签。
    condition_*/field_<field>.npy: float32 场数组 [T,N]。
    condition_*/time.npy: 各工况的时间标记 [T]；根目录 time.npy 取首个工况。
    N、D、E、T 分别为节点数、坐标维度、有向边数和该工况帧数。

处理流程:
    解析网格 → 确定节点顺序 → 按单元连边 → 重排并导出各工况场数据。
    各工况须共享网格和节点集合；INP 节点标签唯一，不处理 include 或实例坐标变换。
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from core.common import load_bundle, load_config
from core.data import finite

# 编辑器运行时的默认配置；相对路径以项目根目录为基准，--config 可覆盖。
CONFIG_FILE = "configs/case1.json"


def read_inp(path):
    """返回节点坐标映射 {标签: 坐标} 和各单元的节点标签列表。"""
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
                # 行末逗号表示连接续行；累积完成后去掉首项单元编号。
                pending.extend(int(value) for value in parts)
                if not line.endswith(","):
                    elements.append(pending[1:])
                    pending = []
    if not nodes or not elements or pending:
        raise ValueError(f"INP 中缺少完整节点或单元：{path}")
    return nodes, elements


def csv_node_ids(columns):
    """按列顺序提取列名中的首组数字作为节点标签，并拒绝重复标签。"""
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
    """统一各工况的节点顺序，构建并校验共享图，导出所选字段。"""
    data = config["data"]
    root, output = Path(data["raw_dir"]), Path(data["graph_dir"])
    sources = [folder / f"{data['field']}.csv" for folder in sorted(root.glob("condition_*")) if folder.is_dir()]
    if not sources:
        raise FileNotFoundError(f"没有找到 condition_*/{data['field']}.csv：{root}")

    # 新图采用首个 CSV 的列顺序；已有图则保留其节点顺序，供不同字段共用。
    nodes, elements = read_inp(Path(data["mesh"]))
    reference = pd.read_csv(sources[0], index_col=0)
    node_ids = csv_node_ids(reference.columns)
    graph_path = output / "graph.pt"
    existing = load_bundle(graph_path) if graph_path.is_file() else None
    if existing is not None:
        if set(node_ids) != set(existing["node_ids"].tolist()):
            raise ValueError("当前字段节点与已有图不同，请使用独立 graph_dir。")
        node_ids = existing["node_ids"].tolist()
    # 有限元标签可能不连续，转换为图计算使用的 0..N-1 下标。
    index = {label: i for i, label in enumerate(node_ids)}

    # 同单元内任意两节点双向连边，集合去重，不额外排序。
    edges = set()
    for element in elements:
        for i, first in enumerate(element):
            for second in element[i + 1:]:
                edges.add((first, second))
                edges.add((second, first))
    # 保留场数据节点集合内的边，并将端点标签映射为图下标。
    mapped = [(index[a], index[b]) for a, b in edges if a in index and b in index]
    if not mapped:
        raise ValueError("对齐 CSV 节点后，图中没有边。")
    graph = {
        "pos": torch.tensor([nodes[label] for label in node_ids], dtype=torch.float32),
        # 每列表示一条有向边：[E,2] → [2,E]。
        "edge_index": torch.tensor(mapped, dtype=torch.long).t().contiguous(),
        "node_ids": torch.tensor(node_ids, dtype=torch.long),
    }
    # 精确比较图张量，边的排列顺序不同也视为不一致。
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
        # 按图节点顺序重排列；此处只转换格式和检查有限值，不做归一化。
        values = finite(table.iloc[:, [lookup[label] for label in node_ids]].to_numpy(dtype=np.float32), source)
        if len(values) == 0:
            raise ValueError(f"场序列为空：{source}")
        folder = output / source.parent.name
        folder.mkdir(exist_ok=True)
        np.save(folder / f"field_{data['field']}.npy", values)

        # 数值索引原样保留；任一索引非有限时，整条轴改用帧编号 × dt。
        # 仅生成时间标记，不重采样场数据，也不校验跨工况时间轴一致性。
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
