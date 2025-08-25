import pandas as pd
import numpy
import torch
import os
import config
from tqdm import tqdm
from nas_201_api import NASBench201API 

def format_nats_bench_to_ADL(original_arch_str) -> str:
    cleaned_str = original_arch_str.strip('|')
    ops_with_separators = cleaned_str.split('|')
    ops = [op for op in ops_with_separators if op != '+']
    """
    More readable format(adjustable)
    """
    formatted_ops = []
    for op in ops:
        # print(f"DEBUG: Processing op string: '{op}'")
        if '~' in op:
            op_name, input_node = op.split('~')
        else:
            op_name = op
            input_node = '-1' 
        formatted_ops.append(f"(op:{op_name}, in:{input_node})")
        
    return f"cell=({', '.join(formatted_ops)})"

def parse():
    if not os.path.exists(config.NATS_BENCH_PATH):
        print(f"ERROR: NATS-Bench database file not found at: {config.NATS_BENCH_PATH}")
        return pd.DataFrame()
    
    # requiring sepcific torch edition
    api = NASBench201API(config.NATS_BENCH_PATH, verbose=False)
    
    all_arch_data = []
    num_archs = len(api)

    for arch_index in tqdm(range(num_archs), desc="Parsing Architectures"):
        original_str = api.arch(arch_index)
        adl_str = format_nats_bench_to_ADL(original_str)

        for dataset in ['cifar10', 'cifar100', 'ImageNet16-120']:
            # performance metric (200 epochs)
            performance_info = api.get_more_info(arch_index, dataset, hp='200')

            # cost metric
            cost_info = api.get_cost_info(arch_index, dataset)

            arch_info = {
                'uid': f'nats-tss-{arch_index}-{dataset}', # unique ID
                'source_benchmark': 'nats-bench-tss',
                'arch_id_source': arch_index,
                'dataset_source': dataset,
                'unified_text_description': adl_str,
                
                # --- Performance Metrics ---
                'true_final_train_accuracy': performance_info.get('train-accuracy', -1),
                'true_final_val_accuracy': performance_info.get('valid-accuracy', -1),
                'true_final_test_accuracy': performance_info.get('test-accuracy', -1),
                
                # --- Cost Metrics ---
                'params': cost_info.get('params', -1),
                'flops': cost_info.get('flops', -1),
                'latency': cost_info.get('latency', -1),
                'training_time': performance_info.get('train-all-time', -1),
                # adding more featrues for training....
                # NOTE: There's no zero-cost metrics in these dataset.
                # NOTE: missing true valiation accuracy value in cifar10 (no big deal)
            }
            all_arch_data.append(arch_info)
    print(f"Finished parsing {len(all_arch_data)} records from NATS-Bench TSS.")
    return pd.DataFrame(all_arch_data)

        


