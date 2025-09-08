NAS_BENCH_PATH = "data/NAS-Bench-201-v1_1-096897.pth"

# NAS-Bench-201 的 operator 集合（固定 5 個）
OPERATORS = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]

# 訓練超參數
BATCH_SIZE = 64
LR = 0.001
EPOCHS = 20
