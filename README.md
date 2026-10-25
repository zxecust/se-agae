# SE-AGAE

基于图神经网络自编码器与 LSTM 的瞬态物理场降阶模型，用于物理场的压缩、重构和工况驱动预测。

自编码器从网格与场数据中学习低维空间表示，LSTM 根据工况序列预测潜变量，再通过解码器还原物理场。

## 主要功能

- **图数据构建**：从有限元网格和物理场数据构建图，并对齐节点顺序。
- **自编码器训练**：使用 GraphSAGE 和注意力池化学习潜变量，通过危险区域增强损失关注关键节点。
- **时序模型训练**：使用 LSTM 学习工况序列到潜变量序列的映射。
- **模型预测**：根据输入工况生成潜变量及物理场预测结果。

## 基本流程

```text
空间降阶：物理场 → 图自编码器 → 潜变量 → 重构物理场
时序学习：工况序列 → LSTM → 潜变量序列
模型预测：新工况序列 → LSTM → 解码器 → 预测物理场
```

## 项目结构

```text
se-agae/
├── build_graph.py        # 图数据构建入口
├── train_ae.py           # 自编码器训练入口
├── train_lstm.py         # LSTM 训练入口
├── predict.py            # 模型预测入口
├── core/                 # 共享计算模块
│   ├── models.py         # 网络结构
│   ├── losses.py         # 损失函数
│   ├── data.py           # 数据读取与组织
│   └── common.py         # 配置与训练辅助函数
├── configs/              # 案例配置
├── requirements.txt      # Python 依赖
└── README.md
```

数据和计算输出默认分别存放于 `data/` 和 `outputs/`，不纳入版本管理。

## 环境依赖

- Python 3.10 或更新版本
- NumPy、pandas、PyTorch、PyTorch Geometric

```bash
python -m pip install -r requirements.txt
```

## 基本使用

1. 准备网格、物理场和对应的工况序列，并在 `configs/` 中设置数据路径与训练参数。
2. 在构图和训练入口顶部设置 `CONFIG_FILE`，选择案例配置。
3. 按顺序执行四个入口：

```bash
python build_graph.py
python train_ae.py
python train_lstm.py
python predict.py
```

预测前，在 `predict.py` 顶部设置模型路径 `MODEL_FILE`、工况路径 `CONDITION_FILE` 和输出目录 `OUTPUT_DIR`。

也可以在编辑器中打开入口文件，选择安装了上述依赖的 Python 解释器，直接点击运行整个文件。
命令行参数可覆盖入口顶部的默认设置。

## 当前状态

项目处于开发与代码梳理阶段，目前围绕图构建、自编码器训练、LSTM 训练和预测流程逐步完善。
