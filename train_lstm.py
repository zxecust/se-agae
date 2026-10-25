"""阶段二：固定 AE，以工况序列监督 LSTM 的潜变量预测。"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from core.common import fit, load_bundle, load_config, setup
from core.data import ConditionSequences, collate_sequences, condition_features, condition_path
from core.losses import latent_loss
from core.models import LSTMLatentPredictor

# 在编辑器中点击“运行 Python 文件”时使用；命令行 --config 可覆盖。
CONFIG_FILE = "configs/case1.json"


def train(config):
    device = setup(config)
    output = Path(config["data"]["output_dir"])
    ae = load_bundle(output / "autoencoder.pt")
    if ae["config"]["data"]["field"] != config["data"]["field"]:
        raise ValueError("配置字段与 AE 字段不同，请使用独立的 output_dir。")
    pairs = {}
    for name, latent in ae["latents"].items():
        condition = condition_features(condition_path(config, name), config["features"])
        if len(condition) != len(latent):
            raise ValueError(f"{name} 工况与潜变量帧数不同：{len(condition)} / {len(latent)}")
        pairs[name] = (condition, latent)
    split = ae["split"]  # 两个阶段严格共用同一份工况划分。
    train_pairs = [pairs[name] for name in split["train"]]
    condition = np.concatenate([x for x, _ in train_pairs]).astype(np.float64)
    cond_mean = condition.mean(axis=0).astype(np.float32)
    cond_std = condition.std(axis=0).clip(1e-8).astype(np.float32)
    # 沿用旧项目：标准化仅用于损失；网络输出仍在原始潜空间。
    latent = np.concatenate([z for _, z in train_pairs])
    z_mean = torch.from_numpy(latent.mean(axis=0)).to(device)[None, None]
    z_std = torch.from_numpy(latent.std(axis=0).clip(1e-8)).to(device)[None, None]
    predictor = config["predictor"]
    loaders = [DataLoader(
        ConditionSequences([pairs[name] for name in split[key]], cond_mean, cond_std),
        batch_size=predictor["batch_size"], shuffle=(key == "train"), collate_fn=collate_sequences,
    ) for key in ("train", "val")]
    latent_dim = ae["config"]["model"]["latent_dim"]
    model = LSTMLatentPredictor(len(cond_mean), latent_dim, predictor).to(device)

    def step(batch):
        x, y, mask, lengths = batch
        prediction = model(x.to(device), lengths)
        loss = latent_loss(prediction, y.to(device), mask.to(device), z_mean, z_std, predictor)
        return loss, 1  # 沿用旧项目的逐 batch 平均记录方式。

    print(f"LSTM: device={device}, features={len(cond_mean)}, latent_dim={latent_dim}")
    fit(model, *loaders, step, predictor, output / "lstm_history.csv")
    # 预测文件自包含配套 decoder、图与归一化参数，重新训练 AE 不会使旧预测器错配。
    decoder_state = {key.removeprefix("decoder."): value for key, value in ae["state_dict"].items()
                     if key.startswith("decoder.")}
    bundle = {
        "model_config": ae["config"]["model"], "predictor_config": predictor,
        "feature_config": config["features"], "field": config["data"]["field"],
        "predictor_state": {key: value.cpu() for key, value in model.state_dict().items()},
        "decoder_state": decoder_state, "graph": ae["graph"],
        "condition_mean": cond_mean, "condition_std": cond_std,
        "field_mean": ae["field_mean"], "field_std": ae["field_std"],
        "latent_mean": z_mean.cpu(), "latent_std": z_std.cpu(), "split": split,
    }
    torch.save(bundle, output / "rom.pt")
    print(f"预测模型已保存：{output / 'rom.pt'}")


if __name__ == "__main__":
    train(load_config(CONFIG_FILE))
