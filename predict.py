"""使用所选 ROM 和新工况预测物理场，分别保存场数组、潜变量和执行记录

输入:
    --config 指定目录配置；selection.lstm_run 或 --model 指定配套 ROM
    --condition 指定与训练特征列一致的工况 CSV

输出:
    paths.pred_dir/<prediction_id>/<工况文件名>/<field>.npy: 原始单位场数组 [T,N]
    paths.latent_dir/pred/<prediction_id>/<工况文件名>/z.npy: 预测潜变量 [T,L]
    runs/<case>/<field>/pred/<prediction_id>/run.json: 来源模型和产物位置
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

# 留空 MODEL_FILE 时使用配置中明确选择的 LSTM 训练编号
CONFIG_FILE = "configs/case_hp_rotor.json"
MODEL_FILE = ""
CONDITION_FILE = "D:/ABAQUS/model/HTC_Turbine_Rotor/data/HP_Rotor/conditions/condition_CSCYC_01.csv"
OUTPUT_DIR = ""  # 留空时使用 paths.pred_dir，非空时覆盖预测数据根目录
DEVICE = "auto"
BATCH_SIZE = 20


def predict(config, condition_csv, device, batch_size=20, model_file=None, prediction_id=None, export_csv=False):
    """按独立预测编号保存结果，返回原始单位的物理场数组"""
    if batch_size < 1:
        raise ValueError("batch_size 必须为正数。")
    if model_file:
        target = project_path(model_file)
    else:
        source = selected_run(config, "lstm")
        target = resolve_reference(config, source["artifacts"]["model"])
    saved = load_bundle(target)
    if saved["case"] != config["case"] or saved["field"] != config["data"]["field"] or saved["stage"] != "lstm":
        raise ValueError("ROM 与当前案例、物理场或模型阶段不一致。")
    if not model_file and saved["run_id"] != source["run_id"]:
        raise ValueError("ROM 与所选 LSTM 训练记录不一致。")
    graph = saved["graph"]
    condition_csv = project_path(condition_csv)
    condition = condition_features(condition_csv, saved["feature_config"])
    normalized = (condition - saved["condition_mean"]) / (saved["condition_std"] + 1e-8)
    sequence = torch.from_numpy(normalized.astype(np.float32))[None].to(device)
    predictor = LSTMLatentPredictor(condition.shape[1], saved["model_config"]["latent_dim"],
                                    saved["predictor_config"]).to(device)
    decoder = GNNDecoder(graph["pos"].shape[1], saved["model_config"]).to(device)
    predictor.load_state_dict(saved["predictor_state"])
    decoder.load_state_dict(saved["decoder_state"])
    predictor.eval()
    decoder.eval()
    paths = storage_paths(config)
    with run_session(config, "pred", prediction_id) as run:
        # 来源编号用于关联模型；实际输入路径额外保留，便于追溯本次执行
        run["source_lstm_run_id"] = saved["run_id"]
        run["source_ae_run_id"] = saved["source_ae_run_id"]
        run["sources"] = {"model_file": str(target), "condition_csv": str(condition_csv)}
        name = condition_csv.stem
        field_dir = paths["pred"] / run["run_id"] / name
        latent_dir = paths["latent"] / "pred" / run["run_id"] / name
        field_dir.mkdir(parents=True, exist_ok=False)
        latent_dir.mkdir(parents=True, exist_ok=False)
        field = np.empty((len(condition), len(graph["pos"])), dtype=np.float32)
        with torch.no_grad():
            latent = predictor(sequence).squeeze(0)
            for start in range(0, len(latent), batch_size):
                z = latent[start:start + batch_size]
                pos, edges, batch = batch_graph(graph, len(z), device)
                decoded = decoder(z, pos, edges, batch).reshape(len(z), -1).cpu().numpy()
                field[start:start + len(z)] = decoded * saved["field_std"] + saved["field_mean"]
        np.save(latent_dir / "z.npy", latent.cpu().numpy())
        np.save(field_dir / f"{saved['field']}.npy", field)
        # CSV 是可选查看格式，不默认重复保存一份完整物理场
        if export_csv:
            table = pd.DataFrame(field, columns=[f"Node-{int(label)}" for label in graph["node_ids"]])
            table.index.name = "frame_index"
            table.to_csv(field_dir / f"{saved['field']}.csv")
        run["artifacts"] = {"field": reference(config, "pred", field_dir / f"{saved['field']}.npy"),
                            "latent": reference(config, "latent", latent_dir / "z.npy")}
        run["shape"] = list(field.shape)
        print(f"预测完成：id={run['run_id']}, field={field.shape}, saved to {field_dir}")
    return field


def main(argv=None):
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
    config = read_config(args.config)
    if args.output:
        config["paths"]["pred_dir"] = str(project_path(args.output))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    predict(config, args.condition, device, args.batch_size, args.model, args.prediction_id,
            args.csv or config["export"]["pred_csv"])


if __name__ == "__main__":
    main()
