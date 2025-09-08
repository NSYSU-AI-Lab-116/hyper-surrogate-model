# data/preprocess.py
import numpy as np
from data.config import OPERATORS

def archstr_to_onehot(arch_str):
    """
    將 arch_str 轉成 one-hot 向量
    e.g. "|nor_conv_3x3~0|+|skip_connect~0|..." -> [0,1,0,0,0, ...]
    """
    tokens = arch_str.replace("|", "").split("+")
    features = []
    for token in tokens:
        op = token.split("~")[0]
        onehot = [1 if op == o else 0 for o in OPERATORS]
        features.extend(onehot)
    return np.array(features, dtype=np.float32)

def preprocess_dataframe(df):
    X = np.stack([archstr_to_onehot(a) for a in df["arch"]])
    y = df["acc"].values
    return X, y
