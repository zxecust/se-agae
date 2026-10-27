# SE-AGAE

基于图神经网络自编码器与 LSTM 的瞬态物理场降阶模型，用于物理场的压缩、重构和工况驱动预测。

自编码器从网格与场数据中学习低维空间表示，LSTM 根据工况序列预测潜变量，再通过配套解码器还原物理场。

## 基本流程

```text
build_graph.py：INP + 原始场 CSV → 共享图和对齐的场数组
train_ae.py：场数组 → 最佳 AE 模型 + 配套潜变量 + 评估记录
train_lstm.py：工况序列 + 指定 AE 的潜变量 → 最佳 ROM 模型
predict.py：新工况 + 指定 ROM → 预测潜变量和物理场
```

## 环境依赖

使用 Python 3.10 或更新版本，安装 NumPy、pandas、PyTorch 和 PyTorch Geometric：

```powershell
python -m pip install -r requirements.txt
```

## 路径配置

在案例 JSON 中配置三类根目录：

- `paths.data_root`：大型数据根目录，本机为 `D:/ROM`
- `paths.models_root`：模型根目录，默认相对项目根目录的 `models`
- `paths.runs_root`：运行记录根目录，默认相对项目根目录的 `runs`

大型数据仅分为四类：`graph`、`latent`、`recon`、`pred`。各类目录通过 `paths.graph_dir`、`paths.latent_dir`、`paths.recon_dir`、`paths.pred_dir` 分别设置，相对 `data_root` 解析，也可指定绝对路径。换电脑时修改根目录配置即可。

这些目录支持 `{case}` 和 `{field}` 占位符，例如 `"graph_dir": "{case}/graph/{field}"`，切换案例或物理场时自动使用相应标签。

INP 和原始 CSV 保留在 ABAQUS 工作路径，使用 `data.mesh`、`data.raw_dir` 独立指定，不复制进大型数据根目录。工况特征表通过 `data.condition_dir` 和 `data.condition_pattern` 指定。相对输入路径统一相对项目根目录；案例中的占位路径须改为实际文件位置。

以 `case_hip_rotor` 的 `stress` 字段为例：

```text
项目根目录/
├── models/case_hip_rotor/stress/
│   ├── ae/<ae_run>/autoencoder.pt
│   └── lstm/<lstm_run>/rom.pt
└── runs/case_hip_rotor/stress/
    ├── ae/summary.csv
    ├── ae/<ae_run>/{history.csv, metrics.csv, run.json}
    ├── lstm/summary.csv
    ├── lstm/<lstm_run>/{history.csv, metrics.csv, run.json}
    └── pred/<prediction_id>/run.json

D:/ROM/case_hip_rotor/
├── graph/stress/
│   ├── graph.pt
│   └── condition_*/stress.npy
├── latent/stress/
│   ├── ae/<ae_run>/condition_*/z.npy
│   └── pred/<prediction_id>/<工况文件名>/z.npy
├── recon/stress/ae/<ae_run>/condition_*/stress.npy
└── pred/stress/<prediction_id>/<工况文件名>/stress.npy
```

图中保留原始节点标签；不额外生成 `node_ids.npy` 和 `time.npy`。物理场数组直接使用字段名，如 `stress.npy`、`temp.npy`，形状为 `[T,N]`。

## 训练与模型选择

1. 设置案例输入路径、目录配置和训练参数，运行构图及 AE 训练：

```powershell
python build_graph.py --config configs/case1.json
python train_ae.py --config configs/case1.json
```

2. AE 每次训练自动创建独立编号并打印位置。将选定编号填写到配置的 `selection.ae_run`，再训练 LSTM：

```powershell
python train_lstm.py --config configs/case1.json
```

3. 将选定的 LSTM 编号填写到 `selection.lstm_run`，输入新工况进行预测：

```powershell
python predict.py --config configs/case1.json --condition C:/ABAQUS/conditions/new_condition.csv
```

也可用 `--model` 显式指定 ROM 文件。`--output` 覆盖预测数据根目录，程序仍追加独立预测编号和工况目录；`--prediction-id` 可指定未使用的新编号。

模型文件保留结构参数、权重、图和归一化参数；AE 潜变量独立放在 `latent` 中。LSTM 只读取明确选择且已完成的 AE 记录和配套潜变量，不自动使用最新模型。ROM 自带对应解码器，因此后续重训 AE 不会改变既有 ROM。

`case_hip_rotor.json` 仍只填写了构图所需输入，训练参数和工况特征需在开始训练前补齐。入口顶部的 `CONFIG_FILE` 用于编辑器直接运行时选择配置，命令行 `--config` 可覆盖。

## 查看结果与按需导出

- `history.csv`：逐轮训练与验证损失
- `metrics.csv`：逐工况指标；AE 使用原始单位 RMSE、MAE，LSTM 使用原始潜空间 RMSE、MAE
- `run.json`：实际配置、输入来源、数据划分、最佳轮次、运行状态和产物引用
- `summary.csv`：同案例、同物理场和同模型阶段下，每次训练一行；失败训练保留状态，不可作为后续输入

AE 汇总 RMSE、MAE 按全部验证工况的所有帧和节点统计；LSTM 汇总指标按验证潜变量元素统计，不将各工况 RMSE 简单平均。不同划分或不同损失口径的训练不直接用 `val_loss` 排名。

配置 `export.recon_conditions` 指定需要保存完整 AE 重构场的工况，默认空列表仅计算误差。预测默认保存 NPY；`export.pred_csv=true` 或命令行 `--csv` 才额外导出 CSV。

训练数据和模型目录不纳入版本管理。旧模型包及旧目录没有自动回退或迁移逻辑；本次命名与目录调整后，已有数据需按新配置明确迁移或重新生成。
