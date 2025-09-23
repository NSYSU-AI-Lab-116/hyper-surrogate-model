"""
Trainable Language Model

This module provides a clean, trainable language model based on Gemma-3-270m-it
with LoRA support for efficient fine-tuning.
"""

import os
import warnings

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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
from peft import LoraConfig, get_peft_model, TaskType
import json

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
        use_lora: Optional[bool] | None = None,
        lora_config: Optional[Dict[str, Any]] = None,
        task_type: Literal["generation", "regression", "both"] = "regression",
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
        self.use_lora = use_lora if use_lora is not None else config.model.use_lora
        self.task_type = task_type
        self.num_outputs = num_outputs
        logger.info(f"Using LoRA: {self.use_lora}")
        
        # Log configuration being used
        logger.debug(f"Generation config: max_new_tokens={config.generation.max_new_tokens}, "
                    f"temperature={config.generation.temperature}, top_k={config.generation.top_k}, "
                    f"top_p={config.generation.top_p}")
    
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
            "task_type": self.task_type,
            "num_outputs": self.num_outputs,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_ratio": trainable_params / total_params * 100,
            "numerical_head_parameters": numerical_params,
        }