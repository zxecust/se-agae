"""产物管理模块：统一产物路径、运行编号、模型选择和运行记录"""

import csv
import json
import time
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from core.common import project_path


# 路径管理类函数
def storage_paths(config):
    """返回各类目录的绝对路径字典，仅解析路径，不创建目录"""
    paths = config["paths"]
    # 项目内根目录相对项目位置解析，也允许配置绝对路径
    data_root = project_path(paths["data_root"])
    roots = {name: project_path(paths[f"{name}_root"]) for name in ("models", "runs")}
    # 大型数据目录支持案例、字段占位符；相对路径归入 data_root，绝对路径直接使用
    for name in ("graph", "latent", "recon", "pred"):
        value = paths[f"{name}_dir"].format(case=config["case"], field=config["data"]["field"])
        roots[name] = (data_root / value).resolve()
    return roots


def reference(config, root, path):
    """用逻辑根目录和相对路径记录产物，迁移后由当前配置重新定位"""
    # root 是 models、latent 等逻辑名称，待记录路径必须位于该根目录内
    relative = Path(path).resolve().relative_to(storage_paths(config)[root])
    # 统一使用正斜杠保存相对路径，读取时由当前环境重新解析
    return {"root": root, "path": relative.as_posix()}


def resolve_reference(config, value):
    """将记录中的 root、path 引用还原为当前配置下的完整路径"""
    return storage_paths(config)[value["root"]] / value["path"]


def model_path(config, stage, run_id):
    """按案例、物理场、训练阶段和编号定位模型文件"""
    filename = {"ae": "autoencoder.pt", "lstm": "rom.pt"}[stage]
    return storage_paths(config)["models"] / config["case"] / config["data"]["field"] / stage / run_id / filename


def record_dir(config, stage, run_id):
    """按案例、物理场、阶段和编号定位运行记录目录"""
    return storage_paths(config)["runs"] / config["case"] / config["data"]["field"] / stage / run_id


# 数据写入类函数
def write_json(path, values):
    """覆盖写入 JSON，保留中文并拒绝 NaN、Inf 数值"""
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(values, stream, indent=2, ensure_ascii=False, allow_nan=False)


def write_rows(path, rows):
    """以首行字典的键为列名，覆盖写入非空记录列表"""
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# 运行记录选择函数
def selected_run(config, stage):
    """仅加载配置明确选择且已完成的训练，不自动寻找最新版本"""
    run_id = config["selection"][f"{stage}_run"]
    if not run_id:
        raise ValueError(f"请在 selection.{stage}_run 中指定训练编号。")
    with (record_dir(config, stage, run_id) / "run.json").open(encoding="utf-8") as stream:
        run = json.load(stream)
    # 返回的是训练记录字典；模型和数组由调用方根据 artifacts 引用继续读取
    identity = (config["case"], config["data"]["field"], stage, run_id)
    if tuple(run[key] for key in ("case", "field", "stage", "run_id")) != identity or run["status"] != "completed":
        raise ValueError("训练记录未完成，或与当前案例、物理场及训练编号不一致。")
    return run


# 运行编号生成函数
def make_run_id():
    """按本地年月日、时分秒和微秒生成编号，可在数据准备前显示"""
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")


# 运行记录管理函数
@contextmanager
def run_session(config, stage, run_id=None):
    """通过 with 管理一次训练或预测的记录，不覆盖已有编号"""
    # 1. 创建独立目录，保存初始运行记录
    started = datetime.now().astimezone()
    run_id = run_id or make_run_id()
    if not run_id or run_id in (".", "..") or any(char in run_id for char in '/\\:'):
        raise ValueError("运行编号必须是单个目录名。")
    # 编号必须是单个目录名，exist_ok=False 阻止复用历史运行目录
    directory = record_dir(config, stage, run_id)
    directory.mkdir(parents=True, exist_ok=False)
    # 预测只记录来源模型，不创建新的模型保存目录
    if stage != "pred":
        model_path(config, stage, run_id).parent.mkdir(parents=True, exist_ok=False)
    # 深拷贝实际配置，避免调用方后续修改配置影响本次记录
    run = {"run_id": run_id, "case": config["case"], "field": config["data"]["field"],
            "stage": stage, "status": "running", "started_at": started.isoformat(),
            "config": deepcopy(config), "artifacts": {}}
    path = directory / "run.json"
    write_json(path, run)
    # 单调时钟用于计时，避免系统时间调整影响耗时统计
    clock = time.monotonic()

    # 2. 交由调用方执行，按正常结束或普通异常更新状态
    try:
        # with ... as run 获取同一字典，可补充 training、metrics、artifacts 等字段
        yield run
    except Exception as error:
        run.update(status="failed", error=str(error))
        raise  # 保留异常传播，交由上层处理
    else:
        run["status"] = "completed"
    finally:
        # 3. 写入结束记录，为训练阶段追加一行历史摘要
        run.update(finished_at=datetime.now().astimezone().isoformat(),
                    elapsed_seconds=round(time.monotonic() - clock, 3))
        write_json(path, run)
        if stage != "pred":
            # 物理场误差与潜空间误差分列保存，不能跨空间直接比较
            summary = directory.parent / "summary.csv"
            columns = ["run_id", "status", "started_at", "elapsed_seconds", "best_epoch", "best_val_loss",
                        "val_rmse", "val_mae", "val_latent_rmse", "val_latent_mae", "source_ae_run_id"]
            # 展平最佳轮次和评估指标；缺失字段留空，兼容尚未完成的训练记录
            row = {**run, **run.get("training", {}), **run.get("metrics", {})}
            exists = summary.is_file()
            with summary.open("a", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                # 新建汇总文件时写入表头，后续运行只追加记录
                if not exists:
                    writer.writeheader()
                writer.writerow({key: row.get(key, "") for key in columns})
