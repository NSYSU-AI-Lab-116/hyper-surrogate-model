import time
from pathlib import Path
import torch

SRC = Path(r"C:/aiLab/data/NAS-Bench-201-v1_1-096897.pth")
DST = Path(r"C:/aiLab/data/nasbench201_clean.pth")

# 若仍想讓程式自動回退找 OneDrive 的來源，可以加上這個候補路徑：
CANDIDATES = [
    SRC,
    Path(r"C:/Users/ASUS/OneDrive/Desktop/aiLab/hyper-surrogate-model/data/NAS-Bench-201-v1_1-096897.pth"),
]

def find_existing(paths):
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError("找不到來源檔，請確認路徑。")

def allowlist_numpy_scalars():
    try:
        import numpy as np
        from numpy.core.multiarray import scalar as np_scalar
        # 依錯誤訊息加入 numpy scalar；也一併加入常見 dtype/class
        torch.serialization.add_safe_globals([np_scalar, np.dtype, np.generic])
    except Exception as e:
        print(f"[warn] allowlist 設定時出現問題：{e}（會照常嘗試載入）")

def safe_torch_load(path: Path):
    t0 = time.time()
    # 1) 直接安全讀
    try:
        print(f"[load] weights_only=True → {path}")
        obj = torch.load(path, map_location="cpu", weights_only=True)
        print(f"[load] ok (weights_only=True) in {time.time()-t0:.2f}s")
        return obj
    except Exception as e:
        print(f"[load] weights_only=True 失敗：{e}")

    try:
        allowlist_numpy_scalars()
        t1 = time.time()
        print(f"[load] retry with allowlist (weights_only=True) → {path}")
        obj = torch.load(path, map_location="cpu", weights_only=True)
        print(f"[load] ok (allowlisted, weights_only=True) in {time.time()-t1:.2f}s")
        return obj
    except Exception as e:
        print(f"[load] retry 仍失敗：{e}")

    print("[warn] 將回退至 weights_only=False（不安全）。請**確認來源可信**後再繼續。")
    t2 = time.time()
    obj = torch.load(path, map_location="cpu", weights_only=False)
    print(f"[load] ok (weights_only=False) in {time.time()-t2:.2f}s")
    return obj

def main():
    src = find_existing(CANDIDATES)
    print(f"[info] source: {src}")
    print(f"[info] target: {DST}")

    data = safe_torch_load(src)

    # （可選）裁剪只保留必要鍵，檔案更小、讀取更快：
    # 若 data 是 dict，可只留你需要的欄位，例如：
    # needed_keys = ["meta_archs", "total_archs", "arch2infos", "evaluated_indexes"]
    # if isinstance(data, dict):
    #     data = {k: data[k] for k in needed_keys if k in data}

    t0 = time.time()
    DST.parent.mkdir(parents=True, exist_ok=True)
    print(f"[save] writing clean file → {DST}")
    torch.save(data, DST)  # 這裡存出的會是純 tensors/dicts，之後讀取很快
    print(f"[save] done in {time.time()-t0:.2f}s")
    print("[ok] conversion finished. 之後請在程式中改讀 nasbench201_clean.pth")

if __name__ == "__main__":
    main()
