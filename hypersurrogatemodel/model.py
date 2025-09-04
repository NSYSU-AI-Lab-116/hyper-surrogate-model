"""
Trainable Language Model

This module provides a clean, trainable language model based on Gemma-3-270m-it
with LoRA support for efficient fine-tuning.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
)
from peft import LoraConfig, get_peft_model, TaskType

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
        template: str = """structured""",
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        """
        Generate text based on the input prompt.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling or greedy decoding
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text string
        """
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        # Move inputs to the same device as the model
        device = next(self.model.parameters()).device
        inputs = inputs.to(device)
        prompt = """Generate a structured response in JSON format:
                    Input: "Analyze the weather in Tokyo"
                    Output format: {"location": "", "analysis": "", "temperature": "", "conditions": ""}

                    Input: "{your_input}"
                    Output:
                """
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        # Decode only the new tokens
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.shape[-1]:], 
            skip_special_tokens=True
        )
        
        return generated_text
    
    def get_tokenizer(self) -> PreTrainedTokenizer:
        """Get the tokenizer for this model."""
        return self.tokenizer
    
    def save_model(self, save_path: str) -> None:
        """
        Save the model.
        
        Args:
            save_path: Path to save the model
        """
        if self.use_lora:
            # Save LoRA adapters
            self.model.save_pretrained(save_path)
        else:
            # Save full model
            self.model.save_pretrained(save_path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)
    
    @classmethod
    def load_model(cls, load_path: str, **kwargs) -> "TrainableLLM":
        """
        Load a saved model.
        
        Args:
            load_path: Path to the saved model
            **kwargs: Additional arguments for model initialization
            
        Returns:
            TrainableLLM: Loaded model instance
        """
        # Create instance (this will load base model and setup LoRA if specified)
        model = cls(**kwargs)
        
        # Load the fine-tuned weights
        try:
            model.model = AutoModelForCausalLM.from_pretrained(load_path)
            model.tokenizer = AutoTokenizer.from_pretrained(load_path)
        except:
            # If loading fails, keep the initialized model
            logger.warning(f"Could not load model from {load_path}, using base model")
        
        return model
    
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
