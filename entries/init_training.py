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

logger = Logger("Pipelined-runner")
torch.set_float32_matmul_precision('high')


def train_with_dataset(model_path, dataset_path, epochs, batch_size, learning_rate):
    with open(dataset_path, 'r') as f:
        train_data = json.load(f)
    train_length = len(train_data) 
    logger.info(f"loaded {train_length} training samples")
    
    batches_per_epoch = (len(train_data) + batch_size - 1) // batch_size
    total_batches = epochs * batches_per_epoch
    
    logger.info(f"total train of {total_batches} batches ({epochs} epochs × {batches_per_epoch} batches/epoch)")
    
    model = TrainableLLM(
        task_type="regression",  # numerical prediction 
        num_outputs=1,
        use_lora=True,
        base_model_name=model_path,
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
                text = sample['text']
                answer = float(sample['answer'])
                texts.append(text)
                targets.append(answer)
            
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
            numerical_targets = torch.tensor(targets, device=device).unsqueeze(1)
            
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
            if global_batch_count % 5 == 0:
                torch.cuda.empty_cache()
            
            total_loss += loss.item()
            num_batches += 1
            global_batch_count += 1
            
            current_avg_loss = total_loss / num_batches
            
            # toverall progress bar
            total_pbar.update(len(batch_data))
            total_pbar.set_postfix({
                'Avg Loss': f'{current_avg_loss:6.3f}',
            })
            
            # batch progress bar 
            batch_pbar.set_postfix({
                'Batch loss': f'{current_avg_loss:6.3f}',
            })
            
        
        batch_pbar.close()
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        torch.cuda.empty_cache()
        
        # epoch progress bar 
        epoch_pbar.set_postfix({
            'Epoch Avg Loss': f'{avg_loss:.4f}',
        })

    logger.success("Training complete")
    epoch_pbar.close()
    total_pbar.close()
    

    try:
        model.save_model(save_training_state=True,
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
    train_with_dataset(
        model_path=config.model.pretrained_model,
        dataset_path=config.dataset.dataset_path,
        epochs=config.training.num_epochs, 
        batch_size=config.training.batch_size, 
        learning_rate=config.training.learning_rate)