import json
from typing import Optional, Dict, Any, Literal
import torch
from torch.optim import AdamW
from tqdm import tqdm


from hypersurrogatemodel import TrainableLLM, Logger
from hypersurrogatemodel.config import config

logger = Logger("Pipelined-runner")
torch.set_float32_matmul_precision('high')

def training(model_path:str, dataset_path:str, epochs:int, batch_size:int, learning_rate:float):
    with open(dataset_path, 'r') as f:
        train_data = json.load(f)
    train_length = len(train_data) 
    logger.info(f"loaded {train_length} training samples")
    
    batches_per_epoch = (len(train_data) + batch_size - 1) // batch_size
    total_batches = epochs * batches_per_epoch
    
    logger.info(f"total train of {total_batches} batches ({epochs} epochs × {batches_per_epoch} batches/epoch)")
    
    model = TrainableLLM(load_type="from_pretrained")
    logger.info(str(model.get_model_info()))
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    device = model.device
    model.train()
    
    logger.info(f"開始訓練 {epochs} 個 epochs，批次大小: {batch_size}")
    
    # epoch progress bar
    epoch_pbar = tqdm(
        range(epochs), 
        desc="Epochs",
        unit="epoch",
        position=0,
        leave=True
    )
    
    # overall progress bar
    total_pbar = tqdm(
        total=train_length*epochs,
        desc="Overall",
        unit="samples",
        position=1,
        leave=True
    )
    
    global_batch_count = 0
    
    for epoch in epoch_pbar:
        total_loss = 0.0
        num_batches = 0
        
        batch_indices = list(range(0, len(train_data), batch_size))
        batch_pbar = tqdm(
            batch_indices,
            desc="Batch",
            unit="batch",
            position=2,
            leave=False
        )
        input_template = "uid:{uid}, structure:{unified_text_description}"
        
        for i in batch_pbar:
            batch_data = train_data[i:i+batch_size]
            texts = []
            targets = []
            
            for sample in batch_data:
                text = input_template.format(uid=sample["uid"], unified_text_description=sample["unified_text_description"])
                answer = float(sample["true_acc"])
                texts.append(text)
                targets.append(answer)
            
            # Tokenize
            inputs = model.tokenizer(
                texts, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=512,
                return_attention_mask=True
            )
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            numerical_targets = torch.tensor(targets, device=device).unsqueeze(1).float()
            
            optimizer.zero_grad()

            outputs = model(
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
            if global_batch_count % 5 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            total_loss += loss.item()
            num_batches += 1
            global_batch_count += 1
            
            current_avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            
            total_pbar.update(len(batch_data))
            total_pbar.set_postfix({
                'Avg Loss': f'{current_avg_loss:6.3f}',
            })
            
            batch_pbar.set_postfix({
                'Batch loss': f'{current_avg_loss:6.3f}',
            })
            
        
        batch_pbar.close()
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        epoch_pbar.set_postfix({
            'Epoch Avg Loss': f'{avg_loss:.4f}',
        })

    logger.success("Training complete")
    epoch_pbar.close()
    total_pbar.close()
    
    try:
        model.save_model(
                        model_name=model_path,
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
    training(
        model_path=config.model.pretrained_model,    # type: ignore
        dataset_path=config.dataset.train_data_path, # type: ignore
        epochs=config.training.num_epochs,           # type: ignore
        batch_size=config.training.batch_size,       # type: ignore
        learning_rate=config.training.learning_rate  # type: ignore
    ) 