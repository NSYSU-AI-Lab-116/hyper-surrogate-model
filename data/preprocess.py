# data/preprocess.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple
from data.config import OPERATORS

def archstr_to_onehot(arch_str: str) -> np.ndarray:
    tokens = str(arch_str).split('|')
    op2idx = {op: i for i, op in enumerate(OPERATORS)}
    onehot = np.zeros((len(tokens), len(OPERATORS)), dtype=np.float32)
    for r, tok in enumerate(tokens):
        if tok in op2idx:
            onehot[r, op2idx[tok]] = 1.0
    return onehot.flatten()

def _select_numeric_features(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    cand = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cand.append(c)
    return cand

def preprocess_dataframe(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if "y" not in df.columns:
        raise KeyError("找不到欄位 'y'，請確認 parser.load_dataset 已建立 'y'。")

    # arch to one-hot
    if "arch" in df.columns and df["arch"].notna().any():
        X = np.stack([archstr_to_onehot(a) for a in df["arch"].fillna("")])
        y = df["y"].to_numpy(dtype=np.float32)
        feature_names = [f"arch_onehot_{i}" for i in range(X.shape[1])]
        print(f"[INFO] Using 'arch' one-hot feature → X.shape={X.shape}")
        return X, y, feature_names

    leakage_cols = {
        "true_final_train_accuracy",
        "true_final_val_accuracy",
        "true_final_test_accuracy", 
        "accuracy", "test_acc" 
    }

    base_exclude = {
        "y", "dataset", "dataset_source",
        "source_benchmark", "unified_text_description",
        "uid", "arch_id_source"
    }

    exclude = base_exclude | leakage_cols
    numeric_cols = _select_numeric_features(df, exclude=list(exclude))

    # optional: 偏好與效能/規模相關的欄位排在前面
    preferred = [c for c in ["params", "flops", "latency", "training_time"] if c in numeric_cols]
    others = [c for c in numeric_cols if c not in preferred]
    feature_cols = preferred + others

    if not feature_cols:
        print("[WARN] 找不到可用數值特徵欄位。")
        #X = np.zeros((len(df), 1), dtype=np.float32)
        #y = df["y"].to_numpy(dtype=np.float32)
        #return X, y, ["const_zero"]

    X = df[feature_cols].fillna(0).to_numpy(dtype=np.float32)
    y = df["y"].to_numpy(dtype=np.float32)
    print(f"[INFO] Columns that used numerical features: {feature_cols} → X.shape={X.shape}")
    return X, y, feature_cols

def preprocess_Xy(df: pd.DataFrame):
    X, y, _ = preprocess_dataframe(df)
    return X, y
