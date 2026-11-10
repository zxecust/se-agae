"""使用所选 ROM 和新工况预测物理场，分别保存场数组、潜变量和执行记录

输入:
    --config 指定案例与目录配置；selection.lstm_run 或 --model 指定配套 ROM
    网络结构、特征规则和归一化参数均以所选 ROM 内保存的内容为准
    --condition 指定与训练特征列一致的工况 CSV

输出:
    paths.pred_dir/<prediction_id>/<工况文件名>/<field>.npy: 原始单位场数组 [T,N]
    paths.latent_dir/pred/<prediction_id>/<工况文件名>/z.npy: 预测潜变量 [T,L]
    runs/<case>/<field>/pred/<prediction_id>/run.json: 来源模型和产物位置
    工况目录名取输入 CSV 的文件名（不含扩展名）；T、N、L 为帧数、节点数和潜变量维度
    仅显式启用 CSV 导出时额外保存 <field>.csv，新执行编号不覆盖历史结果

处理流程:
    加载指定模型 → 工况归一化 → LSTM 预测潜变量 → 解码并反归一化 → 分类保存
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from core.artifacts import resolve_reference, run_session, selected_run, storage_paths, reference
from core.common import batch_graph, load_bundle, project_path, read_config
from core.data import condition_features
from core.models import GNNDecoder, LSTMLatentPredictor

# 编辑器运行的默认参数可由命令行覆盖；MODEL_FILE 留空时使用 selection.lstm_run
CONFIG_FILE = "configs/case_hp_rotor.json"
MODEL_FILE = ""
CONDITION_FILE = "D:/ABAQUS/model/HTC_Turbine_Rotor/data/HP_Rotor/conditions/condition_CSCYC_01.csv"
OUTPUT_DIR = ""  # 非空时仅覆盖预测物理场目录，潜变量和记录仍使用配置路径
DEVICE = "auto"
# 仅限制解码器每批处理的帧数，LSTM 仍一次处理完整工况序列
BATCH_SIZE = 20


def predict(config, condition_csv, device, batch_size=20, model_file=None, prediction_id=None, export_csv=False):
    """预测单个工况，按独立编号保存结果并返回原始单位数组 [T,N]"""
    # 1. 选择 ROM，核对案例、物理场和模型身份
    if batch_size < 1: # 检查 batch_size 是否有效
        raise ValueError("batch_size 必须为正数。")
    # 显式模型路径优先，否则依据已完成的 LSTM 训练记录定位模型
    if model_file: # 优先使用显式指定的模型文件
        target = project_path(model_file)
    else: # 没有指定模型文件的时候，根据训练记录定位模型
        source = selected_run(config, "lstm")
        target = resolve_reference(config, source["artifacts"]["model"])
    saved = load_bundle(target) # 读取模型包
    # 检查 ROM 的案例、物理场和阶段是否与当前配置一致
    if saved["case"] != config["case"] or saved["field"] != config["data"]["field"] or saved["stage"] != "lstm":
        raise ValueError("ROM 与当前案例、物理场或模型阶段不一致。")
    if not model_file and saved["run_id"] != source["run_id"]:
        raise ValueError("ROM 与所选 LSTM 训练记录不一致。")

    # 2. 按 ROM 中的规则提取工况特征，并复用训练归一化参数
    # ROM 已携带配套 AE 的标准化坐标、边和节点标签，不重复标准化坐标
    graph = saved["graph"]
    condition_csv = project_path(condition_csv)
    condition = condition_features(condition_csv, saved["feature_config"])
    normalized = (condition - saved["condition_mean"]) / (saved["condition_std"] + 1e-8)
    # [T,F] 增加单工况批维为 [1,T,F]，特征列及其顺序由 saved 中的配置决定
    sequence = torch.from_numpy(normalized.astype(np.float32))[None].to(device)

    # 3. 重建预测器与解码器，恢复训练权重并切换为评估模式
    # 预测阶段不需要编码器，也不依赖外部 AE 模型文件
    predictor = LSTMLatentPredictor(condition.shape[1], saved["model_config"]["latent_dim"],
                                    saved["predictor_config"]).to(device)
    decoder = GNNDecoder(graph["pos"].shape[1], saved["model_config"]).to(device)
    predictor.load_state_dict(saved["predictor_state"])
    decoder.load_state_dict(saved["decoder_state"])
    predictor.eval()
    decoder.eval()

    # 4. 创建预测记录，生成潜变量并分批解码物理场
    paths = storage_paths(config)
    # prediction_id 留空时生成新编号，运行状态和耗时在退出上下文时写入记录
    with run_session(config, "pred", prediction_id) as run:
        # 来源编号用于关联模型；实际输入路径额外保留，便于追溯本次执行
        run["source_lstm_run_id"] = saved["run_id"]
        run["source_ae_run_id"] = saved["source_ae_run_id"]
        run["sources"] = {"model_file": str(target), "condition_csv": str(condition_csv)}
        name = condition_csv.stem
        field_dir = paths["pred"] / run["run_id"] / name
        latent_dir = paths["latent"] / "pred" / run["run_id"] / name
        field_dir.mkdir(parents=True, exist_ok=False) # 创建预测物理场目录，若已存在则报错
        latent_dir.mkdir(parents=True, exist_ok=False) # 创建潜变量目录，若已存在则报错
        # CPU 数组保存完整结果 [T,N]，解码过程的设备中间张量按帧分批处理
        field = np.empty((len(condition), len(graph["pos"])), dtype=np.float32)
        # eval 控制网络行为，no_grad 关闭梯度记录；单条完整序列无需补齐 mask
        # 预测器输出 [1,T,L]，解码器输出 [B*N,1]，按帧分批写入 field
        with torch.no_grad():
            # [1,T,L] → [T,L]，输出已处于 AE 原始潜空间，无需潜变量反标准化
            latent = predictor(sequence).squeeze(0)
            for start in range(0, len(latent), batch_size):
                # 当前块帧数 B 可能小于 batch_size，按实际长度构造独立批图
                z = latent[start:start + batch_size]
                pos, edges, batch = batch_graph(graph, len(z), device)
                # 解码 [B*N,1] → [B,N]，列顺序与 ROM 图中的 node_ids 一致
                decoded = decoder(z, pos, edges, batch).reshape(len(z), -1).cpu().numpy()
                # 使用配套 AE 的场值统计量恢复物理单位，并写入对应帧区间
                field[start:start + len(z)] = decoded * saved["field_std"] + saved["field_mean"]

        # 5. 分别保存潜变量、物理场和产物引用，可选导出 CSV
        np.save(latent_dir / "z.npy", latent.cpu().numpy())
        np.save(field_dir / f"{saved['field']}.npy", field)
        # CSV 列名使用原始有限元标签，行索引为帧编号，不代表物理时间
        if export_csv:
            table = pd.DataFrame(field, columns=[f"Node-{int(label)}" for label in graph["node_ids"]])
            table.index.name = "frame_index"
            table.to_csv(field_dir / f"{saved['field']}.csv")
        # 逻辑根目录加相对路径用于迁移定位，实际输入路径已记录在 sources 中
        run["artifacts"] = {"field": reference(config, "pred", field_dir / f"{saved['field']}.npy"),
                            "latent": reference(config, "latent", latent_dir / "z.npy")}
        run["shape"] = list(field.shape)
        print(f"预测完成：id={run['run_id']}, field={field.shape}, saved to {field_dir}")
    return field


def main(argv=None):
    """解析命令行、应用本次预测设置，并启动单工况推理"""
    # 1. 解析参数，未指定的选项采用文件顶部的默认值
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_FILE)
    parser.add_argument("--model", type=Path, default=MODEL_FILE or None, help="显式指定 ROM，否则使用 selection.lstm_run")
    parser.add_argument("--condition", type=Path, default=CONDITION_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR or None, help="覆盖 paths.pred_dir")
    parser.add_argument("--prediction-id", help="指定新的预测编号，已有编号不覆盖")
    parser.add_argument("--csv", action="store_true", help="额外导出 CSV 物理场")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args(argv)

    # 2. 读取配置，应用物理场输出目录覆盖和推理设备设置
    config = read_config(args.config)
    if args.output:
        config["paths"]["pred_dir"] = str(project_path(args.output))
    # 设备由 --device 或顶部 DEVICE 决定，不使用训练配置中的 device
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                            else "cpu" if args.device == "auto" else args.device)

    # 3. 调用预测；命令行 --csv 与配置 export.pred_csv 任一开启即导出 CSV
    predict(config, args.condition, device, args.batch_size, args.model, args.prediction_id,
            args.csv or config["export"]["pred_csv"])


if __name__ == "__main__":
    main()
