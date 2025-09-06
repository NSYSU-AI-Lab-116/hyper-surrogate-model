"""
Trainable Language Model

This module provides a clean, trainable language model based on Gemma-3-270m-it
with LoRA support for efficient fine-tuning.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import os
import json

from .utils import Logger

# Set up logger using utils.Logger
logger = Logger("model")


class TrainableLLM(nn.Module):
    """
    Trainable Language Model for text generation tasks.
    
    This model provides a clean interface for training/fine-tuning a language model
    with LoRA support for parameter-efficient training.
    """
    
    def __init__(
        self,
        base_model_name: str = "google/gemma-3-270m-it",
        use_lora: bool = True,
        lora_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the trainable language model.
        
        Args:
            base_model_name: Name of the base model to load
            use_lora: Whether to use LoRA for parameter-efficient fine-tuning
            lora_config: Custom LoRA configuration dictionary
        """
        super().__init__()
        
        self.base_model_name = base_model_name
        self.use_lora = use_lora
        self.lora_config = lora_config
        
        # Load causal LM for text generation
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            device_map=None,  # Don't use auto device mapping on MPS
        )
        # Move to CPU to avoid MPS issues
        self.model = self.model.to("cpu") # type: ignore
        
        if use_lora:
            self._setup_lora(lora_config)
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # 確保 pad_token 和 eos_token 不同
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            # 添加一個新的 pad token
            self.tokenizer.add_special_tokens({'pad_token': '<PAD>'})
            # 調整模型的 embedding 大小以適應新的 token
            self.model.resize_token_embeddings(len(self.tokenizer))
    
    def _setup_lora(self, lora_config: Optional[Dict[str, Any]] = None) -> None:
        """Set up LoRA configuration for efficient fine-tuning."""
        if lora_config is None:
            lora_config = {
                "task_type": TaskType.CAUSAL_LM,
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            }
        
        self.lora_config = lora_config
        config = LoraConfig(**lora_config)
        self.model = get_peft_model(self.model, config)  # type: ignore
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """
        Forward pass for training.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels for language modeling loss
            **kwargs: Additional arguments
            
        Returns:
            Model outputs including loss if labels provided
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
    
    def generate_text(
        self,
        prompt: str,
        template: str = "structured",
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        """
        Generate text based on the input prompt.
        
        Args:
            prompt: Input text prompt
            template: Template type (not used in current implementation)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling or greedy decoding
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text string
        """
        # 使用 tokenizer 正確編碼並包含 attention_mask
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
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs
            )
        
        # Decode only the new tokens
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:], 
            skip_special_tokens=True
        )
        
        return generated_text
    
    def get_tokenizer(self) -> PreTrainedTokenizer:
        """Get the tokenizer for this model."""
        return self.tokenizer
    
    def save_model(self, save_path: str, save_training_state: bool = False, 
                  optimizer: Optional[torch.optim.Optimizer] = None,
                  scheduler: Optional[Any] = None,
                  epoch: Optional[int] = None,
                  step: Optional[int] = None,
                  loss: Optional[float] = None) -> None:
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
        
        # Save model configuration
        model_config = {
            "base_model_name": self.base_model_name,
            "use_lora": self.use_lora,
            "lora_config": self.lora_config,
        }
        
        with open(os.path.join(save_path, "model_config.json"), "w") as f:
            json.dump(model_config, f, indent=2)
        
        # Save training state if requested
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
                "lora_config": None
            }
            logger.warning(f"No model config found at {config_path}, using defaults")
        
        # Create base model instance
        model = cls(
            base_model_name=model_config["base_model_name"],
            use_lora=model_config["use_lora"],
            lora_config=model_config.get("lora_config")
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
                model.model = AutoModelForCausalLM.from_pretrained(load_path)
                logger.success(f"Full model loaded from {load_path}")
            
            # Load tokenizer
            model.tokenizer = AutoTokenizer.from_pretrained(load_path)
            if model.tokenizer.pad_token is None:
                model.tokenizer.pad_token = model.tokenizer.eos_token
            
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
        
        return {
            "base_model": self.base_model_name,
            "use_lora": self.use_lora,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_ratio": trainable_params / total_params * 100,
        }


# Keep TextGenerationModel as an alias for backward compatibility
TextGenerationModel = TrainableLLM