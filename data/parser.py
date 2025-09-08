import torch
import pandas as pd
from data.config import NAS_BENCH_PATH

def load_nasbench201():
    """
    載入 NAS-Bench-201 v1.1，回傳 DataFrame 給原本 main.py 使用
    欄位包含：
        arch_index, arch_str, dataset, accuracy
    """
    api_data = torch.load(NAS_BENCH_PATH, map_location="cpu")
    arch_list = list(api_data["arch2infos"].keys())
    datasets = ["cifar10", "cifar100", "ImageNet16-120"]

    records = []
    for idx, arch_str in enumerate(arch_list):
        info = api_data["arch2infos"][arch_str]
        for ds in datasets:
            acc = info.get(f"{ds}-test", {}).get("accuracy", None)
            records.append({
                "arch_index": idx,
                "arch_str": arch_str,
                "dataset": ds.lower(),  # 對應 main.py 的 dataset_name
                "accuracy": acc
            })
    df = pd.DataFrame(records)
    df.rename(columns={"arch_str": "arch", "accuracy": "y"}, inplace=True)
    print("NAS-Bench-201 data loaded, total records:", len(df))
    return df
