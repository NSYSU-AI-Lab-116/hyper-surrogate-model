from pathlib import Path
import os
import json
import pandas as pd
import torch
from torch.optim import AdamW
from tqdm import tqdm

from typing import Optional, Dict, Any, Literal
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from hypersurrogatemodel.config import config
from hypersurrogatemodel import (
    TrainableLLM, 
    Logger,
)
logger = Logger("Pipelined-runner")
torch.set_float32_matmul_precision('high')


class ModelWithCustomHead(TrainableLLM):
    def __init__(
        self,
        base_model: Optional[str]| None = None,
        use_lora: Optional[bool] | None = None,
        num_outputs: int = 1,
    ):
        super().__init__()
        
        # Use configuration values if not provided
        self.base_model = base_model or config.model.pretrained_model
        self.use_lora = use_lora if use_lora is not None else config.model.use_lora
        self.num_outputs = num_outputs
        
        logger.info(f"Initializing model: {self.base_model}")
        logger.info(f"Using LoRA: {self.use_lora}")
        
        # Log generation config
        logger.debug(f"Generation config: max_new_tokens={config.generation.max_new_tokens}, "
                    f"temperature={config.generation.temperature}, top_k={config.generation.top_k}, "
                    f"top_p={config.generation.top_p}")
        
        # Load tokenizer early so pad token can be ensured before tokenization
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load base transformer
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model, # type: ignore
            torch_dtype=torch.float32,
            device_map=None,
            trust_remote_code=True,
            attn_implementation='eager',
        )

        self.hidden_size = self.model.config.hidden_size

        if self.use_lora:
            try:
                self._setup_lora()
            except Exception as e:
                logger.error(f"Failed to setup LoRA: {e}")
                raise

        # Device placement
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Additional numerical head for regression tasks
        if self.task_type in ["regression", "both"]:
            mid = max(1, self.hidden_size // 2)
            self.numerical_head = nn.Sequential(
                nn.Linear(self.hidden_size, mid),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(mid, self.num_outputs)
            ).to(self.device)
            logger.info(f"Added numerical output head with {self.num_outputs} outputs")
        else:
            self.numerical_head = None
        
    def forward(
        self, 
        input_ids, 
        attention_mask=None, 
        labels=None, 
        numerical_targets=None, 
        **kwargs
        ):
        """
        Forward pass for training.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            output_hidden_states=True, 
            return_dict=True,
            **kwargs
        )
        
        result = {
            "logits": outputs.logits if hasattr(outputs, 'logits') else None,
            "loss": outputs.loss if (hasattr(outputs, 'loss') and outputs.loss is not None and self.task_type in ["generation", "both"]) else None,
        }
        
        # Numerical prediction
        if self.task_type in ["regression", "both"] and self.numerical_head is not None:
            hidden_states = outputs.hidden_states[-1] if (hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None) else None
            
            if hidden_states is None:
                # Fallback: re-run with output_hidden_states
                with torch.no_grad():
                    model_outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        return_dict=True,
                        **kwargs
                    )
                    hidden_states = model_outputs.hidden_states[-1]
            
            # Pool last token according to attention_mask (safer than fixed -1)
            if attention_mask is not None:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = hidden_states.size(0)
                last_hidden_states = hidden_states[range(batch_size), sequence_lengths]
            else:
                last_hidden_states = hidden_states[:, -1, :]
            
            last_hidden_states = last_hidden_states.to(next(self.numerical_head.parameters()).device)
            numerical_outputs = self.numerical_head(last_hidden_states)
            result["numerical_outputs"] = numerical_outputs
            
            # Calculate numerical loss if targets provided
            if numerical_targets is not None:
                numerical_targets = numerical_targets.to(numerical_outputs.device).float()
                numerical_loss = nn.MSELoss()(numerical_outputs, numerical_targets)
                result["loss"] = numerical_loss if result["loss"] is None else (result["loss"] + numerical_loss)
        
        return result

def train_with_dataset(model_path:str, dataset_path:str, epochs:int, batch_size:int, learning_rate:float):
    with open(dataset_path, 'r') as f:
        train_data = json.load(f)
    train_length = len(train_data) 
    logger.info(f"loaded {train_length} training samples")
    
    batches_per_epoch = (len(train_data) + batch_size - 1) // batch_size
    total_batches = epochs * batches_per_epoch
    
    logger.info(f"total train of {total_batches} batches ({epochs} epochs × {batches_per_epoch} batches/epoch)")
    
    model = ModelWithCustomHead(
        num_outputs=1,
        use_lora=True,
        base_model=model_path
    )
    logger.info(model.get_model_info())
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
    train_with_dataset(
        model_path=config.model.pretrained_model,    # type: ignore
        dataset_path=config.dataset.train_data_path, # type: ignore
        epochs=config.training.num_epochs,           # type: ignore
        batch_size=config.training.batch_size,       # type: ignore
        learning_rate=config.training.learning_rate) # type: ignore