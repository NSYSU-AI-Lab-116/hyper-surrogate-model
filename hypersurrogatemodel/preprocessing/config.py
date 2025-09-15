import os

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Raw data path for input
NATS_BENCH_PATH = os.path.join(root, 'data', 'raw', 'NAS-Bench-201-v1_1-096897.pth')
# TRANSNAS_BENCH_PATH = os.path.join(root, 'data', 'raw', 'transnas-bench_v10141024.pth')

# Processed data path for output
PREPROCESSED_DATA_PATH = os.path.join(root, 'data', 'processed', 'master_dataset.parquet')


