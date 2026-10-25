"""仅使用保存的 ROM 与新工况 CSV 预测物理场，无需真实场或编码器。"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from core.common import batch_graph, load_bundle, project_path
from core.data import condition_features
from core.models import GNNDecoder, LSTMLatentPredictor

# 在编辑器中直接运行本文件时使用；命令行同名选项可覆盖。
MODEL_FILE = "outputs/case1/S_Mises/rom.pt"
CONDITION_FILE = "data/conditions/case1/processed/case1_condition_01.csv"
OUTPUT_DIR = "outputs/predictions/condition_01"
DEVICE = "auto"
BATCH_SIZE = 20


def predict(model_path, condition_csv, output_dir, device, batch_size=20):
    if batch_size < 1:
        raise ValueError("batch_size 必须为正数。")
    saved = load_bundle(model_path)
    graph = saved["graph"]
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
    field = np.empty((len(condition), len(graph["pos"])), dtype=np.float32)
    with torch.no_grad():
        latent = predictor(sequence).squeeze(0)
        for start in range(0, len(latent), batch_size):
            z = latent[start:start + batch_size]
            pos, edges, batch = batch_graph(graph, len(z), device)
            decoded = decoder(z, pos, edges, batch).reshape(len(z), -1).cpu().numpy()
            field[start:start + len(z)] = decoded * saved["field_std"] + saved["field_mean"]
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "pred_z.npy", latent.cpu().numpy())
    np.save(output_dir / f"pred_{saved['field']}.npy", field)
    table = pd.DataFrame(field, columns=[f"Node-{int(label)}" for label in graph["node_ids"]])
    table.index.name = "frame_index"
    table.to_csv(output_dir / f"pred_{saved['field']}.csv")
    print(f"预测完成：field={field.shape}, saved to {output_dir}")
    return field


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_FILE, help="train_lstm.py 保存的 rom.pt")
    parser.add_argument("--condition", type=Path, default=CONDITION_FILE, help="与训练工况列一致的新 CSV")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args(argv)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    predict(project_path(args.model), project_path(args.condition), project_path(args.output), device, args.batch_size)


if __name__ == "__main__":
    main()
