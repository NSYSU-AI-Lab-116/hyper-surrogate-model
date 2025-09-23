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
from .utils import Logger
from hypersurrogatemodel.config import config

logger = Logger("model")
torch.set_float32_matmul_precision('high')

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
            "loss": outputs.loss if (hasattr(outputs, 'loss') and outputs.loss is not None) else None,
        }
        
        # Numerical prediction
        if self.numerical_head is not None:
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
                BATCH_SIZE = hidden_states.size(0)
                last_hidden_states = hidden_states[range(BATCH_SIZE), sequence_lengths]
            else:
                last_hidden_states = hidden_states[:, -1, :]
                logger.warning("Attention mask not provided; using last token for pooling.")
            
            last_hidden_states = last_hidden_states.to(next(self.numerical_head.parameters()).device)  # type: ignore # ensure to set self.numerical_head device
            numerical_outputs = self.numerical_head(last_hidden_states)  # type: ignore
            result["numerical_outputs"] = numerical_outputs
            
            # Calculate numerical loss if targets provided
            if numerical_targets is not None:
                numerical_targets = numerical_targets.to(numerical_outputs.device).float()
                numerical_loss = nn.MSELoss()(numerical_outputs, numerical_targets)
                result["loss"] = numerical_loss if result["loss"] is None else (result["loss"] + numerical_loss)
                
        if numerical_targets is not None and result["loss"] is None:
            logger.error(f"Numerical loss is None despite targets being provided.")
        return result
    
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
        batch_size: Optional[int] = None
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
    
    def trainer(self, ):
        with open(config.dataset.train_data_path, 'r') as f:
            train_data = json.load(f)

        train_length = len(train_data)
        logger.info(f"loaded {train_length} training samples")
        BATCH_SIZE = config.training.batch_size
        EPOCHS = config.training.num_epochs
        
        
        batches_per_epoch = (len(train_data) + BATCH_SIZE - 1) // BATCH_SIZE
        total_batches = EPOCHS * batches_per_epoch

        logger.info(f"total train of {total_batches} batches ({EPOCHS} EPOCHS × {batches_per_epoch} batches/epoch)")

        optimizer = AdamW(self.model.parameters(), lr=config.training.learning_rate)

        # Training progress bars
        epoch_pbar = tqdm(
            range(EPOCHS), 
            desc="Epochs", 
            position=0
        )
        overall_pbar = tqdm(
            total=train_length*EPOCHS,
            desc="Overall",
            position=1
        )

        loss_history = []
        global_batch_count = 0

        for epoch in epoch_pbar:
            total_loss = 0
            num_batches = 0

            batch_pbar = tqdm(
                range(0, len(train_data), BATCH_SIZE), 
                desc="Batches", 
                position=2, 
                leave=False
            )

            input_template = "uid:{uid}, structure:{unified_text_description}"

            for i in batch_pbar:
                batch_data = train_data[i:i+BATCH_SIZE]
                texts = []
                targets = []

                for sample in batch_data:
                    text = input_template.format(uid=sample["uid"], unified_text_description=sample["unified_text_description"])
                    answer = float(sample["true_acc"])
                    texts.append(text)
                    targets.append(answer)

                # Tokenize
                inputs = self.tokenizer(
                    texts, 
                    return_tensors="pt", 
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_attention_mask=True
                )

                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                numerical_targets = torch.tensor(targets, device=self.device).unsqueeze(1)
                optimizer.zero_grad()

                outputs = self(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    numerical_targets=numerical_targets
                )

                loss = outputs.get('loss', None)
                if loss is None:
                    logger.error(f"Failed to get loss, type:{str(loss)}")
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
                    'Epoch': f'{epoch+1}/{EPOCHS}'
                })
                loss_history.append(avg_loss)

            batch_pbar.close()
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            torch.cuda.empty_cache()

            # epoch progress bar 
            epoch_pbar.set_postfix({
                'Epoch Avg Loss': f'{avg_loss:.4f}',
            })
        epoch_pbar.close()
        overall_pbar.close()
        logger.success("Continue training completed!")
        # Save base model
        self.save_model(
                        model_name=self.model_path, # source model name
                        save_training_state=True,
                        optimizer=optimizer,
                        epoch=EPOCHS,
                        dataset_path=config.dataset.train_data_path,
                        batch_size=BATCH_SIZE,
                        )
        np.save(f"{config.hyper.save_basepath}/loss_history.npy", np.array(loss_history))
