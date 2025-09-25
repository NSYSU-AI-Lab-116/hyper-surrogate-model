import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from typing import Optional, Dict, Any, Union, Literal
import json

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import numpy as np
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from .utils import Logger , get_gpu_utilization as gpu_u
from .config import config

logger = Logger("model")
torch.set_float32_matmul_precision('high')


class TrainingDataset(Dataset):
    """Custom dataset for accelerate training"""
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_template = "uid:{uid}, structure:{unified_text_description}"
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        text = self.input_template.format(
            uid=sample["uid"], 
            unified_text_description=sample["unified_text_description"]
        )
        target = float(sample["true_acc"])
        
        # Tokenize on-the-fly (accelerate will handle batching)
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'numerical_target': torch.tensor(target, dtype=torch.float32)
        }


def collate_fn(batch):
    """Custom collate function for batch processing"""
    # Extract components
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    targets = torch.stack([item['numerical_target'] for item in batch]).unsqueeze(1)
    
    # Pad sequences to max length in batch
    max_len = max(len(ids) for ids in input_ids)
    
    # Pad input_ids and attention_masks
    padded_input_ids = []
    padded_attention_masks = []
    
    for ids, mask in zip(input_ids, attention_masks):
        pad_len = max_len - len(ids)
        padded_ids = torch.cat([ids, torch.zeros(pad_len, dtype=torch.long)])
        padded_mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)])
        
        padded_input_ids.append(padded_ids)
        padded_attention_masks.append(padded_mask)
    
    return {
        'input_ids': torch.stack(padded_input_ids),
        'attention_mask': torch.stack(padded_attention_masks),
        'numerical_targets': targets
    }

class TrainableLLM(nn.Module):
    """
    Trainable text generation model
    """
    
    def __init__(
        self,
        load_type: Literal["from_pretrained", "from_saved"],
        use_lora: Optional[bool] | None = None,
        num_outputs: int = 1,
    ):
        """
        Initialize the trainable language model.
        
        Args:
            base_model_name: Name of the base model to load (uses config if None)
            use_lora: Whether to use LoRA for parameter-efficient fine-tuning (uses config if None)
            lora_config: Custom LoRA configuration dictionary (uses config if None)
            num_outputs: Number of numerical outputs (for regression tasks)
        """
        super().__init__()
        self.use_lora = use_lora if use_lora is not None else config.model.use_lora
        self.num_outputs = num_outputs
        self.load_model(load_type=load_type)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device) # type: ignore
        if hasattr(self, 'numerical_head') and self.numerical_head is not None:
            self.numerical_head = self.numerical_head.to(self.device) # type: ignore
        logger.info(f"Model loaded and moved to {self.device}")  
        
        logger.info("Model initialization complete.")   

    def forward(
        self, 
        input_ids, 
        attention_mask=None, 
        numerical_targets=None, 
        **kwargs
    ):

        # Ensure model returns dict + hidden states
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )

        result = {
            "logits": outputs.logits if hasattr(outputs, 'logits') else None,
            "loss": None,  # We'll calculate this properly later
        }
        # Numerical prediction
        if not hasattr(outputs, "hidden_states"):
            raise RuntimeError("Addition hidden layer not working")
        hidden_states = outputs.hidden_states[-1]
        
        # re-run with output_hidden_states
        if hidden_states is None:
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
        if attention_mask is None:
            last_hidden_states = hidden_states[:, -1, :]
            logger.warning("Attention mask not provided; using last token for pooling.")
        else:
            attn = attention_mask.to(hidden_states.device)
            seq_lens = attn.long().sum(dim=1).clamp(min=1) - 1  # shape (B,)
            batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
            last_hidden_states = hidden_states[batch_idx, seq_lens, :]  # shape (B, H)
            
        last_hidden_states = last_hidden_states.to(next(self.numerical_head.parameters()).device)  # type: ignore # ensure to set self.numerical_head device
        numerical_outputs = self.numerical_head(last_hidden_states)  # type: ignore
        result["numerical_outputs"] = numerical_outputs
        
        # Calculate numerical loss if targets provided
        if numerical_targets is not None:
            numerical_targets = numerical_targets.to(numerical_outputs.device).float()
            numerical_loss = self.loss_fn(numerical_outputs, numerical_targets)
            result["loss"] = numerical_loss
        else:
            logger.error(f"Numerical loss is None despite targets being provided.")
        return result
    
    def loss_fn(self, predictions, targets):
        if predictions.shape != targets.shape:
            raise ValueError(f"Predictions shape {predictions.shape} does not match targets shape {targets.shape}")
        return nn.MSELoss()(predictions, targets)
    
    def _setup_lora(self, lora_config: Optional[Dict[str, Any]] = None) -> None:
        
        """Set up LoRA configuration"""
        if lora_config is None:
            # Use configuration from config file
            lora_config = {
                "task_type": TaskType.CAUSAL_LM,
                "r": config.lora.r, # type: ignore
                "lora_alpha": config.lora.lora_alpha,# type: ignore
                "lora_dropout": config.lora.lora_dropout,# type: ignore
                "target_modules": config.lora.target_modules# type: ignore
            }
        self.lora_config = lora_config
        lora_config_obj = LoraConfig(**lora_config)
        self.model = get_peft_model(self.model, lora_config_obj)  # type: ignore

    def load_model(self, load_type: Literal["from_pretrained", "from_saved"]) -> None:
        """load model from pretrained or saved path

        Args:
            mode (str): ["from_pretrained", "from_saved"]
        """
        if load_type == "from_pretrained" and config.model.pretrained_model is not None:
            self.model_path = config.model.pretrained_model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, # type: ignore
                torch_dtype=torch.float32,
                device_map=None,
                trust_remote_code=True,
                attn_implementation='eager',
            )
            logger.info(f"loading model from pretrained: {config.model.pretrained_model}")
            self.hidden_size = self.model.config.hidden_size
            mid = max(1, self.hidden_size // 2)
            self.numerical_head = nn.Sequential(
                nn.Linear(self.hidden_size, mid),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(mid, self.num_outputs)
            )
            logger.info(f"Added numerical output head with {self.num_outputs} outputs")
            
        elif load_type == "from_saved" and config.model.transfer_model_path is not None:
            self.model_path = config.model.transfer_model_path
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, # type: ignore
                torch_dtype=torch.float32,
                device_map=None,
                trust_remote_code=True,
                attn_implementation='eager',
            )
            logger.info(f"loading model from saved: {self.model_path}")
            custom_head_state = torch.load(f'{self.model_path}/numerical_head.pt', map_location='cpu')

            hidden_mid = custom_head_state['0.weight'].shape[0]
            hidden_size = int(self.model.config.hidden_size)

            self.numerical_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_mid),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_mid, 1)
            )
            self.numerical_head.load_state_dict(custom_head_state)
            logger.info(f"Loaded numerical output head with {self.num_outputs} outputs")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        logger.info(f"Tokenizer loaded: {self.tokenizer.__class__.__name__}")
        
        logger.info(f"Using LoRA: {self.use_lora}")
        if self.use_lora:
            self._setup_lora()
               
    def save_model(
        self,
        model_name: Optional[str] = None,
        save_training_state: bool = True, 
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        loss: Optional[float] = None,
        dataset_path: Optional[str] = None,
        batch_size: Optional[int] = None,
        loss_history: Optional[list] = None,
        ) -> None:
        
        """
        Save the model and optionally training state.
        
        Args:
            save_path: Path to save the model
            save_training_state: Whether to save training state (optimizer, scheduler, etc.)
            optimizer: Optimizer state to save
            scheduler: Scheduler state to save
            epoch: Current epoch
            step: Current step
            loss: Current loss
        """
        logger.setFunctionsName("save_model")
        if not hasattr(config.hyper, 'save_basepath') or config.hyper.save_basepath is None:
            raise ValueError("Save path not configured in config.hyper.save_basepath")
        save_path = config.hyper.save_basepath
        
        new_version_dir = config.hyper.new_version_dir
        index_path = config.hyper.index_path
        fs = config.hyper.fs
        logger.step(f"Saving model to {save_path}...")
        import datetime
        os.makedirs(save_path, exist_ok=True)
        fs.append({ # type: ignore
            "version":new_version_dir,
            "save_time": str(datetime.datetime.now()),
            "dataset_path": str(dataset_path),
            "model_path": save_path,
            "config": {
                "base_model_name": model_name,
                "use_lora": self.use_lora,
                "loss": loss,
                "num_outputs": self.num_outputs,
                "max_length": config.dataset.max_length,
                "learning_rate": optimizer.param_groups[0]['lr'] if optimizer is not None else None,
                "epoch": epoch,
                "step": step,
                "batch_size": batch_size,
            }
        })
        with open(index_path, "w+") as f: # type: ignore
            json.dump(fs, f, indent=2)
                
        if self.use_lora:
            # Save LoRA adapters
            self.model.save_pretrained(save_path)
            logger.info(f"LoRA adapters saved to {save_path}")
        else:
            # Save full model
            self.model.save_pretrained(save_path)
            logger.info(f"Full model saved to {save_path}")
        
        # Save tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_path) # type: ignore
        
        
        # Save numerical head if it exists
        if hasattr(self, 'numerical_head') and hasattr(self.numerical_head, 'state_dict'):
            torch.save(
                self.numerical_head.state_dict(),   # type: ignore
                os.path.join(save_path, "numerical_head.pt")
            )
            logger.info(f"Numerical head saved to {save_path}/numerical_head.pt")
        
        if save_training_state:
            training_state = {
                "epoch": epoch,
                "step": step,
                "loss": loss,
            }
            
            if optimizer is not None:
                training_state["optimizer_state_dict"] = optimizer.state_dict()
            
            if scheduler is not None:
                training_state["scheduler_state_dict"] = scheduler.state_dict()
            
            torch.save(training_state, os.path.join(save_path, "training_state.pt"))
            logger.info(f"Training state saved to {save_path}/training_state.pt")
        
        if loss_history is not None:
            np.save(os.path.join(save_path, "loss_history.npy"), np.array(loss_history))
            logger.info(f"Loss history saved to {save_path}/loss_history.npy")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model information
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Add numerical head parameters if present
        if hasattr(self, "numerical_params") and hasattr(self.numerical_head, 'parameters'):
            numerical_params = sum(p.numel() for p in self.numerical_head.parameters()) # type: ignore
            total_params += numerical_params
            trainable_params += numerical_params
        else:
            numerical_params = 0
        
        return {
            "base_model": None if not hasattr(self, 'base_model_name') else self.base_model,
            "use_lora": self.use_lora,
            "num_outputs": self.num_outputs,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_ratio": trainable_params / total_params * 100,
            "numerical_head_parameters": numerical_params,
        }
    
    def acc_trainer(self):
        """Enhanced training method using Accelerate for better performance and scalability"""
        
        # Initialize accelerator
        accelerator = Accelerator(
            mixed_precision="fp16" if torch.cuda.is_available() else "no",
            gradient_accumulation_steps=getattr(config.training, 'gradient_accumulation_steps', 1),
            log_with=None,  # Can be set to "wandb", "tensorboard", etc.
            project_dir=getattr(config.hyper, 'save_basepath', './outputs')
        )
        
        # Load training data
        with open(config.dataset.train_data_path, 'r') as f:
            train_data = json.load(f)

        train_length = len(train_data)
        logger.info(f"Loaded {train_length} training samples")
        
        BATCH_SIZE = config.training.batch_size
        EPOCHS = config.training.num_epochs
        
        # Create dataset and dataloader
        train_dataset = TrainingDataset(train_data, self.tokenizer, max_length=512)
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # Setup optimizer
        params = list(self.model.parameters()) + list(self.numerical_head.parameters())
        optimizer = AdamW(params, lr=config.training.learning_rate)
        
        # Prepare everything with accelerator
        self.model, self.numerical_head, optimizer, train_dataloader = accelerator.prepare(
            self.model, self.numerical_head, optimizer, train_dataloader
        )
        
        # Calculate total steps
        num_update_steps_per_epoch = len(train_dataloader)
        max_train_steps = EPOCHS * num_update_steps_per_epoch
        
        # Log training info
        logger.info(f"***** Running training *****")
        logger.info(f"  Num examples = {train_length}")
        logger.info(f"  Num Epochs = {EPOCHS}")
        logger.info(f"  Batch size per device = {BATCH_SIZE}")
        logger.info(f"  Total train batch size = {BATCH_SIZE * accelerator.num_processes}")
        logger.info(f"  Gradient Accumulation steps = {accelerator.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_train_steps}")
        
        # Training progress tracking
        loss_history = []
        global_step = 0
        
        epoch_bar = tqdm(
            range(EPOCHS), 
            desc="Epochs", 
            leave=True,
            disable=not accelerator.is_local_main_process,
            position=0
        )
        gpu_util = tqdm(
            total=0,
            desc="Device util", 
            leave=True,
            disable=not accelerator.is_local_main_process,
            position=2,
            bar_format='{desc}: {postfix}'
        )
        
        # Training strat
        for epoch in range(EPOCHS):
            self.model.train()
            total_loss = 0
            batch_bar = tqdm(
                range(len(train_dataloader)),
                desc="Batches",
                leave=False,
                position=1
            )
            for step, batch in enumerate(train_dataloader):
                # Forward pass
                outputs = self(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    numerical_targets=batch["numerical_targets"]
                )
                
                loss = outputs.get('loss')
                if loss is None:
                    logger.error(f"Warning: Loss is None at step {step}")
                    continue
                
                accelerator.backward(loss)
                
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.detach().float()
                global_step += 1
                
                # Update progress bar
                if accelerator.is_local_main_process:
                    batch_bar.update(1)
                    avg_loss = total_loss / (step + 1)
                    batch_bar.set_postfix({
                        'loss': f'{avg_loss:.4f}'
                    })
                    loss_history.append(avg_loss.item())
                
                # Clear cache periodically
                if global_step % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gpu_info = gpu_u()
                    gpu_util_percent = gpu_info.get('gpu_utilization_percent', 0)
                    gpu_mem_percent = gpu_info.get('memory_utilization_percent', 0)
                    gpu_used_gb = gpu_info.get('used_memory_gb', 0)
                    gpu_total_gb = gpu_info.get('total_memory_gb', 0)
                    
                    gpu_util.set_postfix_str(
                        f"GPU Util: {gpu_util_percent:.1f}% | "
                        f"GPU Mem: {gpu_mem_percent:.1f}% ({gpu_used_gb:.1f}GB/{gpu_total_gb:.1f}GB)"
                    )
                    
            
            
            if accelerator.is_local_main_process:
                batch_bar.close()
                epoch_bar.update(1)
                epoch_avg_loss = total_loss / len(train_dataloader)
                epoch_bar.set_postfix({
                    'epoch': f'{epoch+1}/{EPOCHS}',
                    'loss': f'{epoch_avg_loss:.4f}'
                })
                
        epoch_bar.close()
        accelerator.wait_for_everyone()
        
        # Save model (only on main process)
        if accelerator.is_local_main_process:
            logger.success("Training completed!")
            
            unwrapped_model = accelerator.unwrap_model(self.model)
            unwrapped_numerical_head = accelerator.unwrap_model(self.numerical_head)
            
            original_model = self.model
            original_head = self.numerical_head
            self.model = unwrapped_model
            self.numerical_head = unwrapped_numerical_head
            
            # Save the model
            # self.save_model(
            #     model_name=self.model_path,
            #     save_training_state=True,
            #     optimizer=optimizer,
            #     epoch=EPOCHS,
            #     dataset_path=config.dataset.train_data_path,
            #     batch_size=BATCH_SIZE,
            #     loss_history=loss_history if loss_history else None
            # )
            
            # Restore original references
            self.model = original_model
            self.numerical_head = original_head
        
        # Final cleanup
        accelerator.end_training()
        logger.info("Accelerated training completed successfully!")
