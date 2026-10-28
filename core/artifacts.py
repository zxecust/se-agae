"""统一模型、训练记录和大型数组的路径及运行编号"""

import csv
import json
import time
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from core.common import project_path


def storage_paths(config):
    """项目内目录相对项目根目录，大型数据目录相对 data_root"""
    paths = config["paths"]
    data_root = project_path(paths["data_root"])
    roots = {name: project_path(paths[f"{name}_root"]) for name in ("models", "runs")}
    for name in ("graph", "latent", "recon", "pred"):
        value = paths[f"{name}_dir"].format(case=config["case"], field=config["data"]["field"])
        roots[name] = (data_root / value).resolve()
    return roots


def reference(config, root, path):
    """用逻辑根目录和相对路径记录产物，迁移后由当前配置重新定位"""
    relative = Path(path).resolve().relative_to(storage_paths(config)[root])
    return {"root": root, "path": relative.as_posix()}


def resolve_reference(config, value):
    return storage_paths(config)[value["root"]] / value["path"]


def model_path(config, stage, run_id):
    filename = {"ae": "autoencoder.pt", "lstm": "rom.pt"}[stage]
    return storage_paths(config)["models"] / config["case"] / config["data"]["field"] / stage / run_id / filename


def record_dir(config, stage, run_id):
    return storage_paths(config)["runs"] / config["case"] / config["data"]["field"] / stage / run_id


def write_json(path, values):
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(values, stream, indent=2, ensure_ascii=False, allow_nan=False)


def write_rows(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def selected_run(config, stage):
    """仅加载配置明确选择且已完成的训练，不自动寻找最新版本"""
    run_id = config["selection"][f"{stage}_run"]
    if not run_id:
        raise ValueError(f"请在 selection.{stage}_run 中指定训练编号。")
    with (record_dir(config, stage, run_id) / "run.json").open(encoding="utf-8") as stream:
        run = json.load(stream)
    identity = (config["case"], config["data"]["field"], stage, run_id)
    if tuple(run[key] for key in ("case", "field", "stage", "run_id")) != identity or run["status"] != "completed":
        raise ValueError("训练记录未完成，或与当前案例、物理场及训练编号不一致。")
    return run


def make_run_id():
    """生成统一运行编号，允许在耗时的数据准备前显示本次编号"""
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")


@contextmanager
def run_session(config, stage, run_id=None):
    """独立记录一次训练或预测；已有编号直接报错，不覆盖历史产物"""
    started = datetime.now().astimezone()
    run_id = run_id or make_run_id()
    if not run_id or run_id in (".", "..") or any(char in run_id for char in '/\\:'):
        raise ValueError("运行编号必须是单个目录名。")
    directory = record_dir(config, stage, run_id)
    directory.mkdir(parents=True, exist_ok=False)
    if stage != "pred":
        model_path(config, stage, run_id).parent.mkdir(parents=True, exist_ok=False)
    run = {"run_id": run_id, "case": config["case"], "field": config["data"]["field"],
           "stage": stage, "status": "running", "started_at": started.isoformat(),
           "config": deepcopy(config), "artifacts": {}}
    path = directory / "run.json"
    write_json(path, run)
    clock = time.monotonic()
    try:
        yield run
    except Exception as error:
        run.update(status="failed", error=str(error))
        raise
    else:
        run["status"] = "completed"
    finally:
        run.update(finished_at=datetime.now().astimezone().isoformat(),
                   elapsed_seconds=round(time.monotonic() - clock, 3))
        write_json(path, run)
        if stage != "pred":
            # 一次训练一行；物理场与潜变量指标使用不同列，避免混淆单位
            summary = directory.parent / "summary.csv"
            columns = ["run_id", "status", "started_at", "elapsed_seconds", "best_epoch", "best_val_loss",
                       "val_rmse", "val_mae", "val_latent_rmse", "val_latent_mae", "source_ae_run_id"]
            row = {**run, **run.get("training", {}), **run.get("metrics", {})}
            exists = summary.is_file()
            with summary.open("a", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                if not exists:
                    writer.writeheader()
                writer.writerow({key: row.get(key, "") for key in columns})
