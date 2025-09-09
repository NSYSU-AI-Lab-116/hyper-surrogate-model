import json
import os

# read config.json
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "..", "config.json")

with open(CONFIG_FILE, "r") as f:
    cfg = json.load(f)

NAS_BENCH_PATH = cfg["NAS_BENCH_PATH"]

OPERATORS = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]

# hyper parameters
BATCH_SIZE = cfg.get("BATCH_SIZE", 64)
LR = cfg.get("LR", 0.001)
EPOCHS = cfg.get("EPOCHS", 20)