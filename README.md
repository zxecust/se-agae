# SE-AGAE 基础版

以 `gnn-based-rom` 为来源整理的单网格、单通道图自编码器降阶模型。
保留图构建、GNN 自编码器训练、LSTM 训练和工况预测四步，优先让数据流和计算过程容易阅读。
这是代码迁移的起点，尚未用原案例数据重新训练验证精度。

## 方法与迁移边界

```text
场 CSV [T,N] + INP → 固定图 (pos, edge_index) + 场数组
场快照 [N,1] → GraphSAGE 编码器 → 图级潜变量 [16] → GraphSAGE 解码器 → 场快照
工况序列 [T,F] → 单向 LSTM → 潜变量序列 [T,16] → 已训练解码器 → 预测场 [T,N]
```

默认方法沿用原 `src/training/train_gnn_lstm.py`：

- GraphSAGE，隐藏维度 96，编码器和解码器各 4 层，潜变量 16 维。
- 编码器输入为场值与标准化节点坐标，使用注意力池化。
- 解码器把图级潜变量广播到每个节点，再与坐标拼接。
- AE 损失为全场 MSE + `danger_weight` × 危险节点及其 BFS 邻域 MSE。
- LSTM 隐藏维度 128、2 层、Dropout 0.1、单向，直接输出原始潜变量。
- LSTM 默认损失为逐维标准化潜变量 MSE + 0.7 × 一阶差分误差 + 0.05 × 二阶差分误差。
- Adam、验证集早停、梯度裁剪；两阶段依次训练，不进行端到端微调。

这里的“白化”仅指逐维标准化，不进行协方差去相关。差分不除以时间步长。
沿用旧主线的开关语义：关闭 `whiten_loss` 时，只计算原始潜变量 MSE；
要在标准化空间禁用某项差分损失，将相应 weight 设为 0。

没有迁移 rotor-blade-rom 新增的时间权重、初始窗口或首帧监督。
首版也不包含 TCN、基线对照、LOOCV、有限元计算、原始工况插值、绘图、损伤计算、断点续训及 AMP。
使用 float32 在 CPU 或 CUDA 上运行；没有 AMP 意味着显存占用和速度可能与旧项目不同。

## 文件与阅读顺序

| 文件 | 作用 |
| --- | --- |
| `configs/case1.json`、`configs/case2.json` | 数据位置、工况特征、模型和训练参数 |
| `build_graph.py` | 从 INP 建图，把字段 CSV 对齐到统一节点顺序 |
| `core/models.py` | 编码器、解码器和 LSTM 的实际计算 |
| `core/losses.py` | 危险区域选择、AE 和时序损失 |
| `core/data.py` | 工况特征、按帧数据集、变长序列补齐 |
| `train_ae.py` | 自编码器训练、重构误差和潜变量导出 |
| `train_lstm.py` | 读取配套潜变量，训练时序预测器 |
| `predict.py` | 新工况到预测场的完整计算 |
| `core/common.py` | 路径配置、固定图的批处理和短训练循环 |

四个入口留在根目录，`core/` 集中保存共享计算实现。入口负责组织任务，核心模块定义计算。
建议先看配置与 `core/models.py`，再依次跟随四个入口读完整流程。

```text
se-agae/
├── build_graph.py
├── train_ae.py
├── train_lstm.py
├── predict.py
├── core/
│   ├── __init__.py
│   ├── models.py
│   ├── losses.py
│   ├── data.py
│   └── common.py
├── configs/
├── README.md
└── requirements.txt
```

## 环境

使用 Python 3.10 或更新版本，以及能导入 PyTorch Geometric 的 Python 环境。
已有 `gnn-based-rom` 环境可直接复用。依赖清单仅列出必要包，不锁定平台/CUDA 版本；
实际 PyTorch 和 PyG 安装应与运行设备匹配。

```bash
python -m pip install -r requirements.txt
```

## 准备数据

不自动复制原项目的大型仿真数据或旧权重。将已有文件放入下面的结构，
或在 JSON 中使用指向已有数据的绝对路径：

```text
data/
  raw/case1/
    case1.inp
    condition_01/S_Mises.csv
    condition_02/S_Mises.csv
    ...
  conditions/case1/processed/
    case1_condition_01.csv
    case1_condition_02.csv
    ...
```

- 场 CSV 第一列为帧标识，其余列为节点；列名可为 `Node-123` 或 `123`。
- 每个工况的字段列会按节点标签重排。所有工况共享同一网格与节点集合。
- 当前 INP 读取器面向节点编号唯一、节点/单元直接写在文件中的单网格。
  多实例重复编号、外部 include 和装配坐标变换需先在上游处理。
- 沿用原项目建边规则：单元内任意两节点双向连接，包括高阶单元的节点。
- case1 工况需含配置中的 7 个原始列。`temperature` 是初始温度；5 个差分列和
  `frame` 由本项目统一计算，得到 13 维输入。不会从字段真值提取工况特征。
- case2 工况需明确含 `hold_time`、`cycle`、`strain_range`，不再猜测或补齐旧格式。
- 场和工况必须逐行对应、帧数一致。不做自动截断或隐式插值。
- 与原项目一致，数值场索引直接保存为时间元数据；非数值 `Step-Frame` 索引使用
  `dt` 生成等间隔时间。构图不修正跨仿真步的时间重置；请先保证行序正确。
  LSTM 使用的是配置里的特征，不会自动把 `time.npy` 当作输入。

`raw_dir`、`mesh`、`graph_dir`、`condition_dir` 和 `output_dir` 的相对路径
均以本项目根目录为基准，与启动命令时所在目录无关。

case1 默认字段为 `S_Mises`；训练温度时改为 `TEMP`，并把 `output_dir` 改为独立目录。
case2 默认字段为原常规入口使用的 `D`；若实际目标是 `dc`，同时修改 `field` 和输出目录。
case2 的四个危险节点取自原项目 case2 专用配置，未继承 case1 的危险节点。
更换网格后必须设置相应的 `danger_nodes`；如不使用危险区域损失，将 `danger_weight` 设为 0。

## 四步运行

### 在编辑器中点击运行

1. 选择已安装 NumPy、pandas、PyTorch 和 PyG 的 Python 解释器。
2. 在 `build_graph.py`、`train_ae.py`、`train_lstm.py` 顶部设置 `CONFIG_FILE`，
   例如 `"configs/case1.json"` 或 `"configs/case2.json"`。三个入口应使用同一案例配置。
3. 按上述顺序打开文件，点击编辑器的“运行 Python 文件”按钮执行整个文件。
4. 在 `predict.py` 顶部设置 `MODEL_FILE`、`CONDITION_FILE`、`OUTPUT_DIR`，再点击运行。
   设备与解码批次大小可通过 `DEVICE` 和 `BATCH_SIZE` 调整。

按钮调用的仍然是普通 Python 脚本；无需手动填写命令行参数或创建编辑器专用配置。
请运行整个入口文件，不要选中片段执行。`core/` 中的模块供入口导入，无需单独运行。
数据文件与训练模型仍需按前述步骤准备好。

### 在命令行中运行

在项目目录执行；显式命令行参数优先于入口顶部的默认值：

```bash
python build_graph.py --config configs/case1.json
python train_ae.py --config configs/case1.json
python train_lstm.py --config configs/case1.json
python predict.py --model outputs/case1/S_Mises/rom.pt --condition data/conditions/case1/processed/case1_condition_01.csv --output outputs/predictions/condition_01
```

case2 将前三步配置改为 `configs/case2.json`，第四步使用对应 `rom.pt` 和工况文件。
从其他目录调用时，脚本位置使用绝对路径。
入口默认值、`--config`、JSON 数据路径以及预测参数的相对路径均以项目根目录为基准，
与编辑器或命令行的当前工作目录无关；绝对路径也可使用。

也可以把 `data` 配置中的原始数据与工况路径指向 `gnn-based-rom` 中已有的对应目录，
将图和训练输出留在本项目内。原项目生成的 PyG `graph.pt` 与本项目保存的字典格式不同，
需重新执行构图；本版不提供旧权重和旧中间文件的兼容加载。

## 输出与数据划分

`output_dir` 保存：

```text
autoencoder.pt        配套 AE 权重、标准化图、场归一化参数、划分及潜变量序列
embedding/*/z.npy     各工况的潜变量 [T,16]，方便独立查看
ae_history.csv        AE 训练和验证损失
ae_metrics.csv        最优 AE 在各工况上的物理量 RMSE / MAE
split.json            训练/验证工况名称
config.json           AE 阶段的完整配置记录
lstm_history.csv      LSTM 训练和验证损失
rom.pt                自包含的 LSTM、配套解码器、图、特征定义与归一化参数
```

LSTM 从 `autoencoder.pt` 读取潜变量，而不是扫描可能残留的独立 `embedding` 文件。
预测从 `rom.pt` 读取所有结构和归一化设置，不需要原字段数据、工况目录或编码器。
预测输出 `pred_z.npy`、`pred_<field>.npy` 和带原节点标签的 `pred_<field>.csv`。
CSV 索引为帧编号，不代表物理时间。

划分方式沿用旧常规主线：按工况、随机种子 42、80%/20%。本版明确让两个阶段复用同一划分，
所有场、工况和潜变量统计量只用训练集计算。验证集用于早停，不是独立测试集。
新工况通过 `predict.py` 单独输入；同网格要求仍然成立。其他网格需要重新构图和训练。

相同输出目录再次训练会覆盖该阶段文件；切换案例或字段应使用独立输出目录。
`rom.pt` 内保存的是训练该 LSTM 时的解码器快照，之后单独重训 AE 不会改变已有 ROM。

## 与旧代码的有意区别

- 去掉全局变量注入、文件格式猜测、缺文件跳过、NaN 置零、长度自动截断与异常重试。
  保留节点集合、有限数值、维度和序列长度等必要检查。
- 模型定义由训练和预测共同导入；预测设置保存在模型中，不需要重复手改结构。
- 每帧按需从 `.npy` 读取，避免旧代码的整场 GPU/CPU 存储回退分支。
- 已存在的共享图固定节点顺序；处理其他字段时仍按该顺序保存。
- 不承诺与原项目逐轮数值相同：数据加载顺序、验证批处理与 float32 执行可能影响优化轨迹。
  核心结构、损失计算和特征构建应分别对照，而不是用相同种子代替验证。

本版用于建立可学习的共同主线。原始案例精度与跨工况泛化仍需使用真实数据重新验证。

## 已完成的验证

迁移时在 CPU 上将新旧实现加载相同参数后，对照了编码器、解码器、变长 LSTM 的输出，
以及危险节点选择、AE 损失、工况衍生特征、时序损失及其梯度；这些对照通过。
另外，用四节点、四工况、不同序列长度的临时数据跑通了 case1 和 case2 的四步流程，
检查了节点列重排、多个字段共用图顺序、独立预测和帧数不一致时报错。
上述是计算与流程验证，不代表真实案例精度验证；CUDA 路径尚未运行。

目录整理后，已从独立工作目录检查四个入口：case1 使用无命令行参数的默认启动方式，
case2 使用命令行覆盖配置与预测参数，两者均用临时合成数据跑通全流程。
这验证了普通 Python 整文件启动所需的导入和路径行为，未实际操作编辑器按钮。
