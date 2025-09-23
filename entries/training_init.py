from pathlib import Path
import os
import json
import pandas as pd
import torch
from torch.optim import AdamW
from tqdm import tqdm
from hypersurrogatemodel import (
    TrainableLLM, 
    Logger,
)
from hypersurrogatemodel.config import config
import numpy as np
import random
from sklearn.model_selection import train_test_split

logger = Logger("Pipelined-runner")
torch.set_float32_matmul_precision('high')

def train_with_dataset(dataset_path, model_path="./saved_model", epochs=6, batch_size=16, learning_rate=1e-5):
    with open(dataset_path, 'r') as f:
        full_data = json.load(f)
        
    logger.info("Performing train-test split (80/20)...")
    train_data, test_data = train_test_split(full_data, test_size=0.2, random_state=20)
    test_data_path = "./data/processed/NAS_bench_201/cifar10_test_set.json"
    with open(test_data_path, 'w') as f:
        json.dump(test_data, f, indent=2)

    train_length = len(train_data) 
    logger.info(f"loaded {train_length} training samples")

    logger.info("Sorting training data by accuracy to prepare for ranking loss...")
    train_data.sort(key=lambda x: float(x['answer']), reverse=True)
    logger.success(f"Sorted {len(train_data)} training samples.")
    
    batches_per_epoch = (len(train_data) + batch_size - 1) // batch_size
    total_batches = epochs * batches_per_epoch
    
    logger.info(f"total train of {total_batches} batches ({epochs} epochs × {batches_per_epoch} batches/epoch)")
    
    model = TrainableLLM(load_type="from_pretrained")
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    device = next(model.model.parameters()).device
    model.train()

    # ranking loss
    loss_fn = torch.nn.MarginRankingLoss(margin=0.1)
    logger.info(f"Using MarginRankingLoss with a margin of {loss_fn.margin}")
    
    logger.info(f"開始訓練 {epochs} 個 epochs，批次大小: {batch_size}")
    
    num_pairs_per_epoch = len(train_data)
    num_batches_per_epoch = (num_pairs_per_epoch + batch_size - 1) // batch_size
    # craeate epoch progress bar
    epoch_pbar = tqdm(
        range(epochs), 
        desc=f"{"Epochs":8}", 
        unit="epoch",
        position=0,
        leave=True
    )
    
    # Create overall progress bar
    total_pbar = tqdm(
        total=num_pairs_per_epoch*epochs,
        desc=f"{"Overall":8}",
        unit="pairs",
        position=1,
        leave=True
    )
    
    for epoch in epoch_pbar:
        total_loss = 0
        num_batches = 0
        
        # create batch progress bar
        batch_pbar = tqdm(
            range(num_batches_per_epoch),
            desc=f"{"Batch":8}",
            unit="batch",
            position=2,
            leave=False
        )
        
        for i in batch_pbar:
            good_prompts = []
            bad_prompts = []
            optimizer.zero_grad()
            
            # matching
            current_batch_size = min(batch_size, num_pairs_per_epoch - (num_batches * batch_size))
            if current_batch_size <= 0: continue

            for _ in range(current_batch_size):
                # 四分位數
                q1 = len(train_data) // 4
                q2 = len(train_data) // 2
                q3 = q1 * 3
                
                # valid interval
                if q1 < 2 or (len(train_data) - q3) < 2:
                    good_sample = random.choice(train_data[:q2])
                    bad_sample = random.choice(train_data[q2:])
                    good_prompts.append(good_sample['text'])
                    bad_prompts.append(bad_sample['text'])
                    continue

                rand_val = random.random()

                # hard pairwise
                if rand_val < 0.4: 
                    idx1 = random.randint(0, q1 - 2)
                    idx2 = random.randint(idx1 + 1, q1 - 1)
                    good_sample = train_data[idx1]
                    bad_sample = train_data[idx2]
                
                # hard pairwise for weak arch.
                elif rand_val < 0.7: 
                    idx1 = random.randint(q3, len(train_data) - 2)
                    idx2 = random.randint(idx1 + 1, len(train_data) - 1)
                    good_sample = train_data[idx1] 
                    bad_sample = train_data[idx2]

                # simple pairwise
                else: 
                    good_sample = random.choice(train_data[:q2])
                    bad_sample = random.choice(train_data[q2:])

                good_prompts.append(good_sample['text'])
                bad_prompts.append(bad_sample['text'])
                
                

            if not good_prompts: continue

            # get model scores
            good_inputs = model.tokenizer(good_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            bad_inputs = model.tokenizer(bad_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)

            good_inputs = {k: v.to(device) for k, v in good_inputs.items()}
            bad_inputs = {k: v.to(device) for k, v in bad_inputs.items()}
            
            optimizer.zero_grad()

            good_scores = model.forward(**good_inputs)["numerical_outputs"]
            bad_scores = model.forward(**bad_inputs)["numerical_outputs"]

            # compute Ranking Loss
            target = torch.ones_like(good_scores)
            loss = loss_fn(good_scores, bad_scores, target)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1

            # Overall progress updated
            total_pbar.update(len(good_prompts))
            total_pbar.set_postfix({
                'Avg Loss': f'{(total_loss / num_batches):.4f}',
                'Epoch': f'{epoch+1}/{epochs}'
            })
            
            # Batch progress
            batch_pbar.set_postfix({'Ranking Loss': f'{(total_loss / num_batches):.4f}'})

        batch_pbar.close()
        
        avg_epoch_loss = total_loss / num_batches if num_batches > 0 else 0
        epoch_pbar.set_postfix({'Epoch Avg Loss': f'{avg_epoch_loss:.4f}'})

    epoch_pbar.close()
    total_pbar.close()
    
    if model_path:
        try:
            model.save_model(save_path=model_path, 
                            save_training_state=True,
                            optimizer=optimizer,
                            epoch=epochs,
                            dataset_path=dataset_path,
                            batch_size=batch_size,
                            )
        except Exception as e:
            logger.error(f"cannot save model: {e}")
    
    logger.success("Finish training")
    return model

if __name__ == "__main__":
    train_with_dataset(dataset_path=Path("./data/processed/NAS_bench_201/cifar10_cleaned.json"),
                       epochs=config.training.num_epochs, batch_size=config.training.batch_size, learning_rate=config.training.learning_rate)