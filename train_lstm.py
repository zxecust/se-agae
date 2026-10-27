"""固定所选 AE，以工况序列监督 LSTM 的潜变量预测

输入:
    selection.ae_run 指定的已完成 AE 训练及其配套 latent 数组
    data.condition_dir 中按 condition_pattern 命名的工况 CSV

输出:
    models/<case>/<field>/lstm/<run_id>/rom.pt: 配套预测器、解码器及归一化参数
    runs/<case>/<field>/lstm/<run_id>/: history.csv、metrics.csv 和 run.json
    同阶段 summary.csv 汇总各次训练，潜变量评估指标不具有物理场单位

处理流程:
    读取所选 AE 和潜变量 → 对齐工况特征 → 训练 LSTM → 分别保存模型与记录
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from core.artifacts import (model_path, record_dir, reference, resolve_reference,
                            run_session, selected_run, write_rows)
from core.common import fit, load_bundle, load_config, setup
from core.data import ConditionSequences, collate_sequences, condition_features, condition_path
from core.losses import latent_loss
from core.models import LSTMLatentPredictor

# 编辑器运行时的默认配置，--config 可覆盖
CONFIG_FILE = "configs/case1.json"


def train(config):
    """读取明确选择的 AE 版本，训练并记录配套的 LSTM 模型"""
    device = setup(config)
    source = selected_run(config, "ae")
    ae = load_bundle(resolve_reference(config, source["artifacts"]["model"]))
    identity = (config["case"], config["data"]["field"], source["run_id"])
    if (ae["case"], ae["field"], ae["run_id"]) != identity or ae["stage"] != "ae":
        raise ValueError("AE 模型与所选训练记录不一致。")
    latent_dir = resolve_reference(config, source["artifacts"]["latent"])
    split = source["split"]  # 两个阶段严格共用同一份工况划分
    pairs = {}
    for name in split["train"] + split["val"]:
        latent = np.load(latent_dir / name / "z.npy")
        condition = condition_features(condition_path(config, name), config["features"])
        if latent.ndim != 2 or latent.shape[1] != ae["model_config"]["latent_dim"]:
            raise ValueError(f"{name} 潜变量形状与所选 AE 不一致。")
        if len(condition) != len(latent):
            raise ValueError(f"{name} 工况与潜变量帧数不同：{len(condition)} / {len(latent)}")
        pairs[name] = (condition, latent)
    train_pairs = [pairs[name] for name in split["train"]]
    condition = np.concatenate([x for x, _ in train_pairs]).astype(np.float64)
    cond_mean = condition.mean(axis=0).astype(np.float32)
    cond_std = condition.std(axis=0).clip(1e-8).astype(np.float32)
    # 潜变量标准化仅用于损失，网络输出仍在原始潜空间
    latent = np.concatenate([z for _, z in train_pairs])
    z_mean = torch.from_numpy(latent.mean(axis=0)).to(device)[None, None]
    z_std = torch.from_numpy(latent.std(axis=0).clip(1e-8)).to(device)[None, None]
    predictor = config["predictor"]
    # batch_size 按完整工况序列计，变长序列由 mask 排除补齐位置
    loaders = [DataLoader(
        ConditionSequences([pairs[name] for name in split[key]], cond_mean, cond_std),
        batch_size=predictor["batch_size"], shuffle=(key == "train"), collate_fn=collate_sequences,
    ) for key in ("train", "val")]
    latent_dim = ae["model_config"]["latent_dim"]
    model = LSTMLatentPredictor(len(cond_mean), latent_dim, predictor).to(device)

    def step(batch):
        x, y, mask, lengths = batch
        prediction = model(x.to(device), lengths)
        loss = latent_loss(prediction, y.to(device), mask.to(device), z_mean, z_std, predictor)
        return loss, 1  # 训练日志保持每个 batch 等权平均的现有口径

    with run_session(config, "lstm") as run:
        run_id = run["run_id"]
        records = record_dir(config, "lstm", run_id)
        run.update(split=split, source_ae_run_id=source["run_id"])
        run["sources"] = {"ae_model": source["artifacts"]["model"], "latent": source["artifacts"]["latent"]}
        print(f"LSTM: run={run_id}, AE={source['run_id']}, device={device}")
        run["training"] = fit(model, *loaders, step, predictor, records / "history.csv")
        # 按工况评估原始潜变量误差，不将其写成物理场 RMSE
        metrics = []
        val_squares, val_absolute, val_count = 0.0, 0.0, 0
        with torch.no_grad():
            for name, pair in pairs.items():
                x, target = ConditionSequences([pair], cond_mean, cond_std)[0]
                predicted = model(x[None].to(device)).squeeze(0).cpu().numpy()
                difference = predicted.astype(np.float64) - target.numpy()
                squares, absolute, count = np.square(difference).sum(), np.abs(difference).sum(), difference.size
                is_train = name in split["train"]
                metrics.append({"condition": name, "split": "train" if is_train else "val",
                                "latent_rmse": float(np.sqrt(squares / count)), "latent_mae": float(absolute / count)})
                if not is_train:
                    val_squares += squares
                    val_absolute += absolute
                    val_count += count
        # 模型自带配套 decoder、图与归一化参数，预测时不依赖外部 AE 文件
        decoder_state = {key.removeprefix("decoder."): value for key, value in ae["state_dict"].items()
                         if key.startswith("decoder.")}
        bundle = {
            "case": config["case"], "field": config["data"]["field"], "stage": "lstm", "run_id": run_id,
            "source_ae_run_id": source["run_id"],
            "model_config": ae["model_config"], "predictor_config": predictor,
            "feature_config": config["features"],
            "predictor_state": {key: value.cpu() for key, value in model.state_dict().items()},
            "decoder_state": decoder_state, "graph": ae["graph"],
            "condition_mean": cond_mean, "condition_std": cond_std,
            "field_mean": ae["field_mean"], "field_std": ae["field_std"],
            "latent_mean": z_mean.cpu(), "latent_std": z_std.cpu(),
        }
        target = model_path(config, "lstm", run_id)
        torch.save(bundle, target)
        write_rows(records / "metrics.csv", metrics)
        run["metrics"] = {"val_latent_rmse": float(np.sqrt(val_squares / val_count)),
                          "val_latent_mae": float(val_absolute / val_count)}
        run["artifacts"] = {"model": reference(config, "models", target)}
        print(f"LSTM 模型：{target}\n训练记录：{records}")
    return run


if __name__ == "__main__":
    train(load_config(CONFIG_FILE))
