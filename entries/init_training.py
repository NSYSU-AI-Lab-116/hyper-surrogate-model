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
from hypersurrogatemodel.dataset import PromptTemplate
import numpy as np
from sklearn.model_selection import train_test_split

logger = Logger("Pipelined-runner")
torch.set_float32_matmul_precision('high')


def train_with_dataset(dataset_path, model_path="./saved_model", epochs=6, batch_size=16, learning_rate=1e-5):
    with open(dataset_path, 'r') as f:
        full_data = json.load(f)
        
    logger.info("Performing train-test split (80/20)...")
    train_data, test_data = train_test_split(full_data, test_size=0.2, random_state=42)
    test_data_path = "./data/processed/NAS_bench_201/cifar10_test_set.json"
    with open(test_data_path, 'w') as f:
        json.dump(test_data, f, indent=2)

    train_length = len(train_data) 
    logger.info(f"loaded {train_length} training samples")

    dataset_filename = Path(dataset_path).stem
    if "cifar100" in dataset_filename:
        dataset_key = "cifar100"
    elif "cifar10" in dataset_filename:
        dataset_key = "cifar10"
    elif "ImageNet16-120" in dataset_filename:
        dataset_key = "imagenet16-120"
    else:
        logger.warning(f"Could not determine dataset key from path: {dataset_path}. Defaulting to 'cifar10'.")
        dataset_key = "cifar10"

    logger.info(f"Determined dataset key as '{dataset_key}' for prompt generation.")
    prompt_generator = PromptTemplate()

    logger.info("Calculating normalization parameters (mean/std)...")
    all_answers = np.array([float(sample['answer']) for sample in train_data])
    answer_mean = np.mean(all_answers)
    answer_std = np.std(all_answers)
    os.makedirs(model_path, exist_ok=True)
    norm_params_path = os.path.join(model_path, "normalization_params.json")
    with open(norm_params_path, 'w') as f:
        json.dump({'mean': answer_mean, 'std': answer_std}, f, indent=2)
    logger.success(f"Normalization parameters saved to {norm_params_path}")
    logger.info(f"  - Mean: {answer_mean:.4f}")
    logger.info(f"  - Std Dev: {answer_std:.4f}")
    
    batches_per_epoch = (len(train_data) + batch_size - 1) // batch_size
    total_batches = epochs * batches_per_epoch
    
    logger.info(f"total train of {total_batches} batches ({epochs} epochs × {batches_per_epoch} batches/epoch)")
    
    model = TrainableLLM(
        task_type="regression",  # numerical prediction 
        num_outputs=1,
        use_lora=True,
        base_model_name="google/gemma-3-270m-it",
    )
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    device = next(model.model.parameters()).device
    model.train()
    
    logger.info(f"開始訓練 {epochs} 個 epochs，批次大小: {batch_size}")
    
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
        total=train_length*epochs,
        desc=f"{"Overall":8}",
        unit="samples",
        position=1,
        leave=True
    )
    
    
    global_batch_count = 0
    
    for epoch in epoch_pbar:
        total_loss = 0
        num_batches = 0
        
        # create batch progress bar
        batch_indices = list(range(0, len(train_data), batch_size))
        batch_pbar = tqdm(
            batch_indices,
            desc=f"{"Batch":8}",
            unit="batch",
            position=2,
            leave=False
        )
        
        for i in batch_pbar:
            batch_data = train_data[i:i+batch_size]
            texts = []
            targets = []
            
            for sample in batch_data:
                original_answer = float(sample['answer'])
                normalized_answer = (original_answer - answer_mean) / answer_std
                full_prompt = prompt_generator.format_prompt(
                dataset_key=dataset_key,
                architecture_string=sample['text'])
                texts.append(full_prompt)
                targets.append(normalized_answer)  # <- Normalization
            
            # Tokenize (with batch)
            inputs = model.tokenizer(
                texts, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=512,
                return_attention_mask=True
            )
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            numerical_targets = torch.tensor(targets, dtype=torch.float32, device=device).unsqueeze(1)
        
            optimizer.zero_grad()

            outputs = model.forward(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                numerical_targets=numerical_targets
            )
            
            loss = outputs.get('loss', None)
            if loss is None:
                logger.error("get loss failed")
                continue
            
            loss.backward()
            optimizer.step()
            if global_batch_count % 10 == 0:
                torch.cuda.empty_cache()
            
            total_loss += loss.item()
            num_batches += 1
            global_batch_count += 1
            
            current_avg_loss = total_loss / num_batches
            
            # toverall progress bar
            total_pbar.update(len(batch_data))
            total_pbar.set_postfix({
                'Avg Loss': f'{current_avg_loss:6.3f}',
                'Epoch': f'{epoch+1}/{epochs}'
            })
            
            # batch progress bar 
            batch_pbar.set_postfix({
                'Batch loss': f'{current_avg_loss:6.3f}',
            })
            
        
        # cloase batch progress bar
        batch_pbar.close()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        torch.cuda.empty_cache()
        
        # epoch progress bar 
        epoch_pbar.set_postfix({
            'Epoch Avg Loss': f'{avg_loss:.4f}',
        })

    
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