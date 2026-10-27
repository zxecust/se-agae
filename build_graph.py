"""将单网格、单字段的逐工况物理场转换为共享图和 NumPy 数组

输入:
    data.mesh: 含节点坐标和单元连接的 Abaqus INP 文件
    data.raw_dir/condition_*/<field>.csv: 第一列为帧标识，不作为场值导出，其余列为节点场值

输出（位于 paths.graph_dir，相对 paths.data_root，已有同名文件会被覆盖）:
    graph.pt: 包含 pos [N,D]、edge_index [2,E]、node_ids [N] 的字典
    condition_*/<field>.npy: float32 场数组 [T,N]
    N、D、E、T 分别为节点数、坐标维度、有向边数和该工况帧数

处理流程:
    解析网格 → 确定节点顺序 → 按单元连边 → 重排并导出各工况场数据
    各工况须共享网格和节点集合；INP 节点标签唯一，不处理 include 或实例坐标变换
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from core.common import load_bundle, load_config
from core.artifacts import storage_paths
from core.data import finite

# 编辑器运行时的默认配置；相对路径以项目根目录为基准，--config 可覆盖
CONFIG_FILE = "configs/case_hp_rotor.json"


def read_inp(path):
    """返回节点坐标映射 {标签: 坐标} 和各单元的节点标签列表"""
    nodes, elements, mode, pending = {}, [], None, []
    with path.open(encoding="utf-8", errors="ignore") as stream: # .inp 文件本质上是文本文件，可以用处理文本的方式读取
        for raw in stream:
            line = raw.strip() # 去除首尾空白字符
            if not line or line.startswith("**"):  # 跳过空行和注释行
                continue
            if line.startswith("*"): # 识别关键字行
                if pending: 
                    # 如果前一单元数据还没有完成，报错避免保存不完整的单元连接关系
                    raise ValueError("INP 单元连接行未完成。")
                keyword = line.split(",")[0].lower() # 按逗号分隔关键字并转化为小写
                mode = keyword if keyword in ("*node", "*element") else None
                continue
            parts = [part.strip() for part in line.split(",") if part.strip()] # 按逗号分隔数据行并去除空白
            if mode == "*node":
                label = int(parts[0])
                if label in nodes: # 节点坐标必须唯一，否则报错
                    raise ValueError("INP 节点编号重复；请提供单个网格，先合并或重编号多个实例。")
                nodes[label] = [float(value) for value in parts[1:]]
            elif mode == "*element":
                # 行末逗号表示连接续行；累积完成后去掉首项单元编号
                pending.extend(int(value) for value in parts)
                if not line.endswith(","):
                    elements.append(pending[1:])
                    pending = []
    if not nodes or not elements or pending:
        raise ValueError(f"INP 中缺少完整节点或单元：{path}")
    return nodes, elements


def csv_node_ids(columns):
    """按列顺序提取列名中的首组数字作为节点标签，并拒绝重复标签"""
    labels = []
    for column in columns:
        match = re.search(r"\d+", str(column)) # 匹配列名中的首个数字序列
        if match is None:
            raise ValueError(f"字段列名无法解析节点编号：{column}")
        labels.append(int(match.group())) # 取出匹配内容，转化为整数并保存
    if len(set(labels)) != len(labels): # 检查是否有重复节点编号；set() 会去重，如果长度不一致说明有重复
        raise ValueError("字段 CSV 含重复节点编号。")
    return labels


def build(config):
    """统一各工况的节点顺序，构建并校验共享图，导出所选字段"""
    # 1. 第一步，读取独立配置的 ABAQUS 输入路径和图输出目录
    data = config["data"] # 获取数据配置字典
    if not data["raw_dir"] or not data["mesh"]:
        raise ValueError("请设置 data.raw_dir 和 data.mesh，指向原始 CSV 与 INP 文件。")
    root, output = Path(data["raw_dir"]), storage_paths(config)["graph"]
    # 按工况目录名排序并拼接字段路径；文件是否存在由后续读取确认
    sources = [folder / f"{data['field']}.csv" for folder in sorted(root.glob("condition_*")) if folder.is_dir()]
    if not sources:
        raise FileNotFoundError(f"没有找到 condition_*/{data['field']}.csv：{root}")

    # mesh 文件就是 Abaqus INP 文件，读取节点坐标和单元连接关系
    nodes, elements = read_inp(Path(data["mesh"])) 
    # 首列作为帧索引，不参与场值导出；其余列确定节点集合与初始顺序
    reference = pd.read_csv(sources[0], index_col=0)
    node_ids = csv_node_ids(reference.columns)
    graph_path = output / "graph.pt"
    # 如果已有图文件存在，加载并检查节点集合是否一致；若不一致则报错，要求使用独立的 graph_dir
    existing = load_bundle(graph_path) if graph_path.is_file() else None
    if existing is not None:
        # 集合比较忽略列顺序；节点集合相同后沿用已有图顺序，供不同字段共用
        if set(node_ids) != set(existing["node_ids"].tolist()):
            raise ValueError("当前字段节点与已有图不同，请使用独立 graph_dir。")
        node_ids = existing["node_ids"].tolist()
    # 有限元标签可能不连续，转换为图计算使用的 0..N-1 下标
    index = {label: i for i, label in enumerate(node_ids)}

    # 2. 第二步，构建图的边集合
    # 同单元内任意两节点双向连边，不限于几何棱边；集合去重，不额外排序
    edges = set()
    for element in elements:
        for i, first in enumerate(element):
            for second in element[i + 1:]:
                edges.add((first, second))
                edges.add((second, first))
    # 保留场数据节点集合内的边，并将端点标签映射为图下标
    mapped = [(index[a], index[b]) for a, b in edges if a in index and b in index]
    if not mapped:
        raise ValueError("对齐 CSV 节点后，图中没有边。")
    # 将坐标、边和节点标签组织为图字典
    graph = {
        # 坐标行顺序与 node_ids 一致；此处保留原始坐标，不做标准化
        "pos": torch.tensor([nodes[label] for label in node_ids], dtype=torch.float32),
        # 每列表示一条有向边：[E,2] → [2,E]
        "edge_index": torch.tensor(mapped, dtype=torch.long).t().contiguous(),
        # 保留原始节点标签，用于从图下标还原有限元节点身份
        "node_ids": torch.tensor(node_ids, dtype=torch.long),
    }
    # 精确比较图张量，边的排列顺序不同也视为不一致
    if existing is not None and any(not torch.equal(graph[key], existing[key]) for key in graph):
        raise ValueError("网格与已有图不同，请使用独立 graph_dir。")
    # 共享图只保存一份，各工况另存场数组；已有同名文件会被覆盖
    output.mkdir(parents=True, exist_ok=True)
    torch.save(graph, output / "graph.pt") # 保存图字典为 PyTorch 文件

    # 3. 第三步，处理各工况的场数据
    for source in sources:
        table = pd.read_csv(source, index_col=0)
        labels = csv_node_ids(table.columns)
        # 各工况可采用不同列顺序，但节点集合必须与共享图完全一致
        if set(labels) != set(node_ids):
            raise ValueError(f"字段的节点集合不一致：{source}")
        # lookup 映射节点标签到当前 CSV 的列下标，供后续按图顺序取列
        lookup = {label: i for i, label in enumerate(labels)}
        # 保留全部帧并按图顺序取列，生成 float32 [T,N]；检查有限值，不做归一化
        values = finite(table.iloc[:, [lookup[label] for label in node_ids]].to_numpy(dtype=np.float32), source)
        if len(values) == 0: # 检查帧数是否为零，避免保存空数组
            raise ValueError(f"场序列为空：{source}")
        folder = output / source.parent.name
        folder.mkdir(exist_ok=True)
        np.save(folder / f"{data['field']}.npy", values)
        print(f"{source.parent.name}: field={values.shape}")
    # 4. 第四步，打印图信息
    print(f"Graph: nodes={len(node_ids)}, edges={len(mapped)}, saved to {output}")


if __name__ == "__main__":
    build(load_config(CONFIG_FILE))
