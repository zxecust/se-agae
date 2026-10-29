"""固定所选 AE，以工况序列监督 LSTM 的潜变量预测

输入:
    selection.ae_run 指定的已完成 AE 训练、模型和配套潜变量数组 [T,L]
    data.condition_dir 中按 condition_pattern 命名的工况 CSV，提取特征后为 [T,F]

输出:
    models/<case>/<field>/lstm/<run_id>/rom.pt: 配套预测器、解码器及归一化参数
    runs/<case>/<field>/lstm/<run_id>/: history.csv、metrics.csv 和 run.json
    同阶段 summary.csv 汇总各次训练，新训练使用独立编号，不覆盖历史结果
    不重复导出已有 AE 潜变量，也不在本阶段生成完整预测场
    T、L、F 分别为工况帧数、潜变量维度和工况特征数；误差在原始潜空间计算

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
CONFIG_FILE = "configs/case_hp_rotor.json"


def train(config):
    """读取明确选择的 AE 版本，训练并记录配套的 LSTM 模型"""
    # 1. 读取所选 AE，并配对工况特征与潜变量
    device = setup(config)
    # 依据训练记录定位配套模型和潜变量，不自动选择最新 AE
    source = selected_run(config, "ae")
    ae = load_bundle(resolve_reference(config, source["artifacts"]["model"]))
    # 同时核对模型包身份，避免训练记录引用了不匹配的模型
    identity = (config["case"], config["data"]["field"], source["run_id"])
    if (ae["case"], ae["field"], ae["run_id"]) != identity or ae["stage"] != "ae":
        raise ValueError("AE 模型与所选训练记录不一致。")
    latent_dir = resolve_reference(config, source["artifacts"]["latent"])
    split = source["split"]  # 两个阶段严格共用同一份工况划分
    latent_dim = ae["model_config"]["latent_dim"]
    # 每个工况保存 (特征 [T,F], 潜变量 [T,L])，按行对应，不插值或重采样
    pairs = {}
    for name in split["train"] + split["val"]:
        latent = np.load(latent_dir / name / "z.npy")
        # 特征按配置选列，可附加逐帧差分和帧编号
        condition = condition_features(condition_path(config, name), config["features"])
        # 校验潜变量维度及其与工况特征的帧数对应关系
        if latent.ndim != 2 or latent.shape[1] != latent_dim:
            raise ValueError(f"{name} 潜变量形状与所选 AE 不一致。")
        if len(condition) != len(latent):
            raise ValueError(f"{name} 工况与潜变量帧数不同：{len(condition)} / {len(latent)}")
        pairs[name] = (condition, latent)

    # 2. 统计归一化参数，准备数据加载器与模型
    # 归一化统计只使用训练工况；拼接帧用于统计，原始工况边界仍由 pairs 保留
    train_pairs = [pairs[name] for name in split["train"]]
    # 特征逐列统计，以 float64 计算后转回 float32，下限避免常量特征除零
    condition = np.concatenate([x for x, _ in train_pairs]).astype(np.float64) # 拼接训练工况的特征 [总帧数,F]
    cond_mean = condition.mean(axis=0).astype(np.float32)
    cond_std = condition.std(axis=0).clip(1e-8).astype(np.float32)
    predictor = config["predictor"]
    # 损失归一化使用训练潜变量统计量 [1,1,L]，广播到 [B,T,L]；输出仍在原始潜空间
    z_mean = z_std = None
    if predictor["whiten_loss"]:
        latent = np.concatenate([z for _, z in train_pairs])
        z_mean = torch.from_numpy(latent.mean(axis=0)).to(device)[None, None]
        z_std = torch.from_numpy(latent.std(axis=0).clip(1e-8)).to(device)[None, None]
    # batch_size 按完整工况序列计；训练集打乱工况顺序，不打乱工况内的帧
    # collate 补齐到本批最长序列：x [B,Tmax,F]、y [B,Tmax,L]、mask [B,Tmax,1]
    loaders = [DataLoader(
        ConditionSequences([pairs[name] for name in split[key]], cond_mean, cond_std),
        batch_size=predictor["batch_size"], shuffle=(key == "train"), collate_fn=collate_sequences,
    ) for key in ("train", "val")]
    model = LSTMLatentPredictor(len(cond_mean), latent_dim, predictor).to(device)

    def step(batch):
        """计算一个 batch 的掩码潜变量损失"""
        x, y, mask, lengths = batch
        # lengths 保留在 CPU 供序列打包使用，LSTM 跳过补齐帧
        prediction = model(x.to(device), lengths)
        # mask 排除补齐位置；开启 whiten_loss 后可附加逐帧一阶、二阶差分损失
        # 差分按帧计算，不除以物理时间间隔；关闭 whiten_loss 时仅计算原始潜变量 MSE
        loss = latent_loss(prediction, y.to(device), mask.to(device), z_mean, z_std, predictor)
        return loss, 1  # 轮次日志按 batch 等权平均

    # 3. 训练 LSTM，并恢复最佳验证权重
    # 本次 LSTM 使用独立编号，同时记录来源 AE 编号和配套输入位置
    with run_session(config, "lstm") as run:
        run_id = run["run_id"]
        records = record_dir(config, "lstm", run_id)
        run.update(split=split, source_ae_run_id=source["run_id"])
        run["sources"] = {"ae_model": source["artifacts"]["model"], "latent": source["artifacts"]["latent"]}
        print(f"LSTM: run={run_id}, AE={source['run_id']}, device={device}")
        # fit 返回前切换到评估模式，并保存逐轮记录
        run["training"] = fit(model, *loaders, step, predictor, records / "history.csv")

        # 4. 逐工况评估原始潜变量预测误差
        metrics = []
        val_squares, val_absolute, val_count = 0.0, 0.0, 0
        with torch.no_grad():
            for name, pair in pairs.items():
                # 单个完整工况没有补齐位置，可直接推理；目标潜变量不做归一化
                x, target = ConditionSequences([pair], cond_mean, cond_std)[0]
                predicted = model(x[None].to(device)).squeeze(0).cpu().numpy()
                # 使用 float64 累计误差；此处不经过 decoder，不计算物理场误差
                difference = predicted.astype(np.float64) - target.numpy()
                squares, absolute, count = np.square(difference).sum(), np.abs(difference).sum(), difference.size
                is_train = name in split["train"]
                metrics.append({"condition": name, "split": "train" if is_train else "val",
                                "latent_rmse": float(np.sqrt(squares / count)), "latent_mae": float(absolute / count)})
                # 汇总全部验证帧和潜变量分量，不对逐工况 RMSE 再做简单平均
                if not is_train:
                    val_squares += squares
                    val_absolute += absolute
                    val_count += count

        # 5. 打包并保存模型、评估指标和训练记录
        # 模型自带配套 decoder、图与归一化参数，预测时不依赖外部 AE 文件
        # 只提取冻结 AE 的解码器权重，去掉前缀以直接加载到 GNNDecoder
        decoder_state = {key.removeprefix("decoder."): value for key, value in ae["state_dict"].items()
                            if key.startswith("decoder.")}
        bundle = {
            # 模型身份、来源和构建网络所需配置
            "case": config["case"], "field": config["data"]["field"], "stage": "lstm", "run_id": run_id,
            "source_ae_run_id": source["run_id"],
            "model_config": ae["model_config"], "predictor_config": predictor,
            "feature_config": config["features"],
            # 最佳 LSTM 权重与固定的 AE 解码器、标准化图
            "predictor_state": {key: value.cpu() for key, value in model.state_dict().items()},
            "decoder_state": decoder_state, "graph": ae["graph"],
            # 预测输入使用训练特征统计量，解码输出使用 AE 场值统计量反归一化
            "condition_mean": cond_mean, "condition_std": cond_std,
            "field_mean": ae["field_mean"], "field_std": ae["field_std"],
        }
        # 模型与日志分别保存，run.json 和 summary.csv 由 run_session 统一写入
        target = model_path(config, "lstm", run_id)
        torch.save(bundle, target)
        write_rows(records / "metrics.csv", metrics)
        run["metrics"] = {"val_latent_rmse": float(np.sqrt(val_squares / val_count)),
                            "val_latent_mae": float(val_absolute / val_count)}
        run["artifacts"] = {"model": reference(config, "models", target)}
        print(f"LSTM model: {target}\nRun logs: {records}")
    return run


if __name__ == "__main__":
    train(load_config(CONFIG_FILE))
