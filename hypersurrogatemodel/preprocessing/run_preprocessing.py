import pandas as pd
from parsers import nats_bench_parser
import config
import os

def main():
    nats_bench_df = nats_bench_parser.parse()
    
    if nats_bench_df.empty:
        print("ERROR: NATS-Bench parsing failed. Aborting.")
        return
    
    final_df = nats_bench_df 
    print(f"\nTotal {len(final_df)} architecture records merged.")
    
    output_dir = os.path.dirname(config.PREPROCESSED_DATA_PATH)
    os.makedirs(output_dir, exist_ok=True)
        
    final_df.to_parquet(config.PREPROCESSED_DATA_PATH)
    print(f"Unified dataset saved to: {config.PREPROCESSED_DATA_PATH}")
    print("--- Finished ---")


if __name__ == '__main__':
    if not os.path.exists('parsers'):
        os.makedirs('parsers')
    if not os.path.exists('parsers/__init__.py'):
        with open('parsers/__init__.py', 'w') as f:
            pass 
    main()