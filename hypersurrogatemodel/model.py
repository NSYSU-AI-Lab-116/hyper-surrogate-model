"""
Trainable Language Model

This module provides a clean, trainable language model based on Gemma-3-270m-it
with LoRA support for efficient fine-tuning.
"""

import os
import warnings

# 設置環境變量和警告過濾以減少不必要的警告
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # 避免多進程問題
# 過濾特定的 transformers 警告
warnings.filterwarnings("ignore", message=".*generation flags.*not valid.*")
warnings.filterwarnings("ignore", message=".*cache_implementation.*")
warnings.filterwarnings("ignore", message=".*early_stopping.*")
warnings.filterwarnings("ignore", message=".*GenerationConfig.*")
warnings.filterwarnings("ignore", message=".*generation_config.*default values.*")
warnings.filterwarnings("ignore", message=".*top_k.*top_p.*")

import torch
import torch.nn as nn

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union, Literal
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import json

from .utils import Logger
from .config import config

# Set up logger using utils.Logger
logger = Logger("model")
torch.set_float32_matmul_precision('high')

class TrainableLLM(nn.Module):
    """
    Trainable text generation model
    """
    
    def __init__(
        self,
        base_model_name: Optional[str] = "google/gemma-3-270m-it",
        use_lora: Optional[bool] = False,
        lora_config: Optional[Dict[str, Any]] = None,
        task_type: Literal["generation", "regression", "both"] = "generation",
        num_outputs: int = 1,
    ):
        """
        Initialize the trainable language model.
        
        Args:
            base_model_name: Name of the base model to load (uses config if None)
            use_lora: Whether to use LoRA for parameter-efficient fine-tuning (uses config if None)
            lora_config: Custom LoRA configuration dictionary (uses config if None)
            task_type: Type of task - "generation", "regression", or "both"
            num_outputs: Number of numerical outputs (for regression tasks)
        """
        super().__init__()
        
        # Use configuration values if not provided
        self.base_model_name = base_model_name or config.model.pretrained_model
        self.use_lora = use_lora if use_lora is not None else config.model.use_lora
        self.lora_config = lora_config
        self.task_type = task_type
        self.num_outputs = num_outputs
        
        logger.info(f"Initializing model: {self.base_model_name}")
        logger.info(f"Using LoRA: {self.use_lora}")
        
        # Log configuration being used
        logger.debug(f"Generation config: max_new_tokens={config.generation.max_new_tokens}, "
                    f"temperature={config.generation.temperature}, top_k={config.generation.top_k}, "
                    f"top_p={config.generation.top_p}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float32,
            device_map=None, 
            trust_remote_code=True,
            attn_implementation='eager',
        )
        self.hidden_size = self.model.config.hidden_size
        
        # Additional layer 
        self.model = self.model.to("cuda") # type: ignore
        if self.task_type in ["regression", "both"]:
            self.numerical_head = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_size // 2, self.num_outputs)
            )
            logger.info(f"Added numerical output head with {self.num_outputs} outputs")
            self.numerical_head = self.numerical_head.to("cuda")
        else:
            self.numerical_head = None
        
        if self.use_lora:
            self._setup_lora(lora_config)
        
        # tokenizer set
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    
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
    
    def forward(self, input_ids, attention_mask=None, labels=None, numerical_targets=None, **kwargs):
        """
        Forward pass for training.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels for language modeling loss
            numerical_targets: Target numerical values for regression (shape: [batch_size, num_outputs])
            **kwargs: Additional arguments
            
        Returns:
            Dict containing model outputs including losses
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels if self.task_type in ["generation", "both"] else None,
            output_hidden_states=True, 
            **kwargs
        )
        
        result = {
            "logits": outputs.logits if hasattr(outputs, 'logits') else None,
            "loss": outputs.loss if (hasattr(outputs, 'loss') and outputs.loss is not None and self.task_type in ["generation", "both"]) else None,
        }
        
        # Numerical prediction
        if self.task_type in ["regression", "both"] and self.numerical_head is not None:
            # Get last hidden state
            hidden_states = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else None
            
            if hidden_states is None: #hidden state un-available
                with torch.no_grad():
                    model_outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        **kwargs
                    )
                    hidden_states = model_outputs.hidden_states[-1]
            
            # Shape: [batch_size, seq_len, hidden_size] -> [batch_size, hidden_size]
            if attention_mask is not None:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = hidden_states.size(0)
                last_hidden_states = hidden_states[range(batch_size), sequence_lengths]
            else:
                last_hidden_states = hidden_states[:, -1, :]
            
            # Numerical prediction outpt
            numerical_outputs = self.numerical_head(last_hidden_states)
            result["numerical_outputs"] = numerical_outputs
            
            # Calculate numerical loss if targets provided
            if numerical_targets is not None:
                numerical_loss = nn.MSELoss()(numerical_outputs, numerical_targets)
                result["loss"] = numerical_loss if result["loss"] is None else (result["loss"] + numerical_loss)
        
        return result
    
    def generate_text(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None,
        **kwargs
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to use sampling
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        if self.task_type not in ["generation", "both"]:
            raise ValueError(f"Text generation not available for task_type='{self.task_type}'")
        
        # Use config values if not provided
        max_new_tokens = max_new_tokens or config.generation.max_new_tokens
        temperature = temperature or config.generation.temperature
        top_k = top_k or config.generation.top_k
        top_p = top_p or config.generation.top_p
        do_sample = do_sample if do_sample is not None else config.generation.do_sample
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        
        # Move inputs to the same device as the model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        self.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        # Decode and return only the new tokens
        input_length = inputs["input_ids"].shape[1]
        generated_text = self.tokenizer.decode(
            outputs[0][input_length:], 
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    def predict_number(
        self,
        prompt: str,
        **kwargs
    ) -> Union[float, torch.Tensor]:
        """
        Predict numerical output from text prompt.
        
        Args:
            prompt: Input text prompt
            **kwargs: Additional arguments
            
        Returns:
            Predicted numerical value(s)
        """
        if self.task_type not in ["regression", "both"]:
            raise ValueError(f"Numerical prediction not available for task_type='{self.task_type}'")
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=512,
            return_attention_mask=True
        )
        
        # Move inputs to the same device as the model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        self.eval()
        with torch.no_grad():
            outputs = self.forward(**inputs)
            numerical_outputs = outputs.get("numerical_outputs")
            
            if numerical_outputs is None:
                raise RuntimeError("No numerical outputs available - model may not be properly configured")
            
            # Return single value if num_outputs=1, otherwise return tensor
            if self.num_outputs == 1:
                return numerical_outputs.squeeze().item()
            else:
                return numerical_outputs.squeeze()
    
    def generate_text_and_number(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        do_sample: Optional[bool] = None,
        **kwargs
    ) -> Dict[str, Union[str, float, torch.Tensor]]:
        """
        Generate both text and numerical output from prompt.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary with 'text' and 'number' keys
        """
        if self.task_type != "both":
            raise ValueError(f"Combined prediction not available for task_type='{self.task_type}'")
        
        result = {}
        
        # Generate text
        result["text"] = self.generate_text(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            **kwargs
        )
        
        # Predict number
        result["number"] = self.predict_number(prompt, **kwargs)
        
        return result
    
    
    def get_tokenizer(self) -> PreTrainedTokenizer:
        """Get the tokenizer for this model."""
        return self.tokenizer
    
    def save_model(
        self, 
        save_path: str, 
        
        save_training_state: bool = True, 
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        loss: Optional[float] = None,
        dataset_path: Optional[str] = None,
        batch_size: Optional[int] = None
        ) -> None:
        
        logger.info(f"Saving model to {save_path}...")
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
        os.makedirs(save_path, exist_ok=True)
        import datetime
        if os.path.exists(os.path.join(save_path, "index.json")):
            index_json_path = os.path.join(save_path, "index.json")
            with open(index_json_path, "r") as f:
                fs = json.load(f)
                
            logger.info(f"Existing versions found: {len(fs)}")
            new_version = f"v_{len(fs)+1}"
            save_path = os.path.join(save_path, new_version)
            os.mkdir(save_path)
            fs.append({
                "version":new_version,
                "save_time": str(datetime.datetime.now()),
                "dataset_path": dataset_path,
                "model_path": save_path,
                "config": {
                    "base_model_name": self.base_model_name,
                    "use_lora": self.use_lora,
                    "loss": loss,
                    "num_outputs": self.num_outputs,
                    "max_length": config.generation.max_new_tokens,
                    "learning_rate": optimizer.param_groups[0]['lr'] if optimizer is not None else None,
                    "epoch": epoch,
                    "step": step,
                    "batch_size": batch_size,
                }
            })
            with open(index_json_path, "w") as f:
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
        self.tokenizer.save_pretrained(save_path)
        
        
        # Save numerical head if it exists
        if self.numerical_head is not None:
            torch.save(
                self.numerical_head.state_dict(), 
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
    
    @classmethod
    def load_model(cls, load_path: str, load_training_state: bool = False, 
                  optimizer: Optional[torch.optim.Optimizer] = None,
                  scheduler: Optional[Any] = None) -> tuple[Union["TrainableLLM", None], Dict[str, Any]]:
        """
        Load a saved model and optionally training state.
        
        Args:
            load_path: Path to the saved model
            load_training_state: Whether to load training state
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            
        Returns:
            Tuple of (TrainableLLM instance, training_state_dict)
        """
        # Load model configuration
        config_path = os.path.join(load_path, "model_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                model_config = json.load(f)
            logger.info(f"Loaded model config from {config_path}")
        else:
            # Fallback to default config
            model_config = {
                "base_model_name": "google/gemma-3-270m-it",
                "use_lora": True,
                "lora_config": None,
                "task_type": "generation",
                "num_outputs": 1,
            }
            logger.warning(f"No model config found at {config_path}, using defaults")
        
        # Create base model instance
        model = cls(
            base_model_name=model_config["base_model_name"],
            use_lora=model_config["use_lora"],
            lora_config=model_config.get("lora_config"),
            task_type=model_config.get("task_type", "generation"),
            num_outputs=model_config.get("num_outputs", 1),
        )
        
        # Load the fine-tuned weights
        try:
            if model.use_lora:
                # Load LoRA adapters
                model.model = PeftModel.from_pretrained(
                    model.model, 
                    load_path
                )
                logger.success(f"LoRA adapters loaded from {load_path}")
            else:
                # Load full model
                model.model = AutoModelForCausalLM.from_pretrained(
                    load_path,
                    attn_implementation='eager'  # Recommended for Gemma3 models
                )
                logger.success(f"Full model loaded from {load_path}")
            
            # Load tokenizer
            model.tokenizer = AutoTokenizer.from_pretrained(load_path)
            if model.tokenizer.pad_token is None:
                model.tokenizer.pad_token = model.tokenizer.eos_token
            
            # Load numerical head if it exists
            numerical_head_path = os.path.join(load_path, "numerical_head.pt")
            if os.path.exists(numerical_head_path) and model.numerical_head is not None:
                model.numerical_head.load_state_dict(
                    torch.load(numerical_head_path, map_location="cpu")
                )
                logger.info(f"Numerical head loaded from {numerical_head_path}")
            
        except Exception as e:
            logger.error(f"Could not load model from {load_path}: {e}")
            return None, {}
        
        # Load training state if requested
        training_state = {}
        if load_training_state:
            training_state_path = os.path.join(load_path, "training_state.pt")
            if os.path.exists(training_state_path):
                training_state = torch.load(training_state_path, map_location="cpu")
                
                # Load optimizer state
                if optimizer is not None and "optimizer_state_dict" in training_state:
                    optimizer.load_state_dict(training_state["optimizer_state_dict"])
                    logger.info("Optimizer state loaded")
                
                # Load scheduler state
                if scheduler is not None and "scheduler_state_dict" in training_state:
                    scheduler.load_state_dict(training_state["scheduler_state_dict"])
                    logger.info("Scheduler state loaded")
                
                logger.success(f"Training state loaded from {training_state_path}")
            else:
                logger.warning(f"No training state found at {training_state_path}")
        
        return model, training_state
    
    def save_checkpoint(self, checkpoint_path: str, 
                       optimizer: torch.optim.Optimizer,
                       scheduler: Optional[Any] = None,
                       epoch: int = 0,
                       step: int = 0,
                       loss: float = 0.0) -> None:
        """
        Save a training checkpoint for resuming training.
        
        Args:
            checkpoint_path: Path to save checkpoint
            optimizer: Current optimizer
            scheduler: Current scheduler (optional)
            epoch: Current epoch
            step: Current step
            loss: Current loss
        """
        self.save_model(
            save_path=checkpoint_path,
            save_training_state=True,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            step=step,
            loss=loss
        )
        logger.success(f"Checkpoint saved to {checkpoint_path}")
    
    @classmethod
    def load_checkpoint(cls, checkpoint_path: str,
                       optimizer: torch.optim.Optimizer,
                       scheduler: Optional[Any] = None) -> tuple[Union["TrainableLLM", None], Dict[str, Any]]:
        """
        Load a training checkpoint to resume training.
        
        Args:
            checkpoint_path: Path to checkpoint
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into (optional)
            
        Returns:
            Tuple of (TrainableLLM instance, training_state_dict)
        """
        model, training_state = cls.load_model(
            load_path=checkpoint_path,
            load_training_state=True,
            optimizer=optimizer,
            scheduler=scheduler
        )
        
        if model is not None:
            logger.success(f"Checkpoint loaded from {checkpoint_path}")
            if training_state:
                logger.info(f"Resuming from epoch {training_state.get('epoch', 0)}, "
                           f"step {training_state.get('step', 0)}, "
                           f"loss {training_state.get('loss', 0.0):.4f}")
        
        return model, training_state
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model information
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Add numerical head parameters if present
        if self.numerical_head is not None:
            numerical_params = sum(p.numel() for p in self.numerical_head.parameters())
            total_params += numerical_params
            trainable_params += numerical_params
        else:
            numerical_params = 0
        
        return {
            "base_model": self.base_model_name,
            "use_lora": self.use_lora,
            "task_type": self.task_type,
            "num_outputs": self.num_outputs,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_ratio": trainable_params / total_params * 100,
            "numerical_head_parameters": numerical_params,
        }