from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
import json 
from torch.optim import AdamW
from tqdm import tqdm
from pathlib import Path
from typing import Optional, List, Dict, Any
from hypersurrogatemodel import (
    Logger, 
    TrainableLLM, 
)
from matplotlib import pyplot as plt
import numpy as np

from hypersurrogatemodel.config import config

logger = Logger("Pipelined-runner")
logger.setFunctionsName("train")

class ModelWithCustomHead(TrainableLLM):
    def __init__(self, base_model, custom_head_path):
        super().__init__()
        # Keep both attributes so TrainableLLM.save_model / get_model_info work
        self.model = base_model

        custom_head_state = torch.load(custom_head_path, map_location='cpu')

        if '0.weight' in custom_head_state:
            hidden_mid = custom_head_state['0.weight'].shape[0]
        else:
            weight_keys = [k for k in custom_head_state.keys() if k.endswith('.weight')]
            if not weight_keys:
                raise RuntimeError("Cannot find weight keys in custom head state dict")
            first_weight = custom_head_state[weight_keys[0]]
            hidden_mid = first_weight.shape[0]

        hidden_size = int(base_model.config.hidden_size)

        self.numerical_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_mid),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_mid, 1)
        )

        self.numerical_head.load_state_dict(custom_head_state)

        if self.use_lora:
            try:
                self._setup_lora()
            except Exception as e:
                logger.error(f"Failed to setup LoRA locally: {e}")
                raise

    def forward(self, input_ids, attention_mask=None, numerical_targets=None, **kwargs):

        # Ensure model returns dict + hidden states
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )

        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            last_hidden_state = outputs.hidden_states[-1]
        else:
            last_hidden_state = outputs.last_hidden_state

        # get last token from attention mask 
        if attention_mask is not None:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.size(0)
            last_token_hidden = last_hidden_state[range(batch_size), sequence_lengths]
        else:
            last_token_hidden = last_hidden_state[:, -1, :]

        numerical_output = self.numerical_head(last_token_hidden)

        result = {
            'logits': outputs.logits,
            'numerical_output': numerical_output,
            'hidden_states': outputs.hidden_states if hasattr(outputs, "hidden_states") else None
        }

        # Calculate loss if targets provided
        if numerical_targets is not None:
            numerical_targets = numerical_targets.to(numerical_output.device)
            loss = nn.MSELoss()(numerical_output, numerical_targets)
            result['loss'] = loss

        return result

def continue_training(
    model_path:str, 
    dataset_path:str,
    epochs:int, 
    batch_size:int, 
    learning_rate:float
):
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Loading pretrained model...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    base_model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation="eager")

    # Build wrapper and attach tokenizer so save_model can persist it
    model = ModelWithCustomHead(base_model, f"{model_path}/numerical_head.pt")
    model.tokenizer = tokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    logger.info(f"Model loaded and moved to {device}")

    # Load training data
    with open(dataset_path, 'r') as f:
        train_data = json.load(f)

    train_length = len(train_data)
    logger.info(f"loaded {train_length} training samples")

    batches_per_epoch = (len(train_data) + batch_size - 1) // batch_size
    total_batches = epochs * batches_per_epoch

    logger.info(f"total train of {total_batches} batches ({epochs} epochs × {batches_per_epoch} batches/epoch)")

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Training progress bars
    epoch_pbar = tqdm(
        range(epochs), 
        desc="Epochs", 
        position=0
    )
    overall_pbar = tqdm(
        total=train_length*epochs,
        desc="Overall",
        position=1
    )

    loss_history = []
    global_batch_count = 0

    for epoch in epoch_pbar:
        total_loss = 0
        num_batches = 0

        batch_pbar = tqdm(
            range(0, len(train_data), batch_size), 
            desc="Batches", 
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
            inputs = tokenizer(
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

            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                numerical_targets=numerical_targets
            )

            loss = outputs.get('loss', None)
            if loss is None:
                logger.error("Failed to get loss")
                continue

            # Backward pass
            loss.backward()
            optimizer.step()

            # clear cache
            if global_batch_count % 5 == 0:
                torch.cuda.empty_cache()


            # Track loss
            total_loss += loss.item()
            num_batches += 1
            global_batch_count += 1

            # Update progress bar
            avg_loss = total_loss / num_batches
            batch_pbar.set_postfix({
                'batch_loss': f'{avg_loss:.4f}',
            })
            overall_pbar.update(len(batch_data))
            overall_pbar.set_postfix({
                'Avg Loss': f'{avg_loss:6.3f}',
                'Epoch': f'{epoch+1}/{epochs}'
            })
            loss_history.append(avg_loss)

        batch_pbar.close()
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        torch.cuda.empty_cache()

        # epoch progress bar 
        epoch_pbar.set_postfix({
            'Epoch Avg Loss': f'{avg_loss:.4f}',
        })
    logger.success("Continue training completed!")
    # Save base model
    model.save_model(
                    model_name=model_path,
                    save_training_state=True,
                    optimizer=optimizer,
                    epoch=epochs,
                    dataset_path=dataset_path,
                    batch_size=batch_size,
                    )
    np.save(f"{config.hyper.save_basepath}/loss_history.npy", np.array(loss_history))

    return model


if __name__ == "__main__":
    # Continue training from saved model
    base_dir = Path(__file__).parent.parent
    continued_model = continue_training(
        model_path=config.model.transfer_model_path, # type: ignore
        dataset_path=config.dataset.train_data_path, # type: ignore 
        epochs=config.training.num_epochs,           # type: ignore
        batch_size=config.training.batch_size,       # type: ignore
        learning_rate=config.training.learning_rate  # type: ignore
    )

