import json, os

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "..", "config.json")
with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    cfg = json.load(f)

NAS_BENCH_PATH = cfg.get("NAS_BENCH_PATH")
MASTER_DATASET_PATH = cfg.get("MASTER_DATASET_PATH")

OPERATORS = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]

BATCH_SIZE = cfg.get("BATCH_SIZE", 64)
LR = cfg.get("LR", 1e-3)
EPOCHS = cfg.get("EPOCHS", 100)
