from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Iterable, Optional
from .config import MASTER_DATASET_PATH

_MASTER = Path(MASTER_DATASET_PATH) if MASTER_DATASET_PATH else Path("data/master_dataset.parquet")
_VALID_DATASETS = {"cifar10", "cifar100", "imagenet16-120"}

def _read_parquet_smart(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except Exception as e_py:
        try:
            return pd.read_parquet(path, engine="fastparquet")
        except Exception as e_fp:
            raise RuntimeError(
                "[ERROR] 無法讀取 parquet。\n"
                f"pyarrow error: {e_py}\nfastparquet error: {e_fp}"
            )


def _ensure_exists(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(
            f"[ERROR] 找不到 {p}。\n"
            "請確認 config.json 的 MASTER_DATASET_PATH 設定正確，或先建立 master_dataset.parquet。"
        )


def load_dataset(
    dataset: Optional[str] = None,
    columns: Optional[Iterable[str]] = None,
    strict: bool = True,
) -> pd.DataFrame:
    
    _ensure_exists(_MASTER)

    df = _read_parquet_smart(_MASTER)

    # 欄名小寫
    df.columns = [c.lower() for c in df.columns]

    rename_map = {
        "data_set": "dataset",
        "dataset_name": "dataset",
        "bench": "dataset",
        "task": "dataset",
        "target": "y",
        "label": "y",
        "acc": "y",
        "accuracy": "y",
        "test_acc": "y",
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

    if columns is not None:
        want = [c.lower() for c in columns]
        df = df[[c for c in want if c in df.columns]]

    def _norm_ds(s: str) -> str:
        s = str(s).strip().lower().replace("_", "-")
        s = s.replace("imagenet-16-120", "imagenet16-120")
        s = s.replace("image-net16-120", "imagenet16-120")
        s = s.replace("cifar-10", "cifar10").replace("cifar-100", "cifar100")
        return s

    if "dataset" not in df.columns:
        if "dataset_source" in df.columns:
            df = df.copy()
            df["dataset"] = df["dataset_source"].map(_norm_ds)
        elif "source_benchmark" in df.columns:
            df = df.copy()
            df["dataset"] = df["source_benchmark"].map(_norm_ds)
        else:
            if dataset is not None:
                df = df.copy()
                df["dataset"] = dataset
                print(f"[WARN] 無 'dataset' / 'dataset_source'，以參數補常數 dataset='{dataset}'（可能導致三個資料集相同）")
            else:
                return df

    # 正規化字串
    df["dataset"] = df["dataset"].map(_norm_ds)

    try:
        vc = df["dataset"].value_counts().to_dict()
        print(f"[INFO] dataset distribution: {vc}")
    except Exception:
        pass

    # 過濾 dataset
    if dataset:
        if strict and dataset not in _VALID_DATASETS:
            raise ValueError(f"[ERROR] dataset 必須是 {_VALID_DATASETS} 之一，收到: {dataset}")
        before = len(df)
        df = df[df["dataset"] == dataset].copy()
        after = len(df)
        if after == 0:
            print(f"[WARN] 依 dataset='{dataset}' 過濾後為 0 筆。")
        elif after == before:
            print(f"[WARN] 過濾前後筆數相同（{before}），可能仍未正確辨識資料集。請檢查 'dataset_source' 原始內容。")

    if "y" not in df.columns:
        if "true_final_test_accuracy" in df.columns:
            df = df.copy()
            df["y"] = df["true_final_test_accuracy"]
        elif "accuracy" in df.columns:
            df = df.copy()
            df["y"] = df["accuracy"]
        else:
            print("[WARN] 找不到可映射到 'y' 的欄位。")

    # print(f"Master dataset loaded from '{_MASTER}'. rows={len(df)} cols={list(df.columns)}")
    return df
