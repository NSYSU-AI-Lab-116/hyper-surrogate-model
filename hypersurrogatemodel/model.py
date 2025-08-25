"""
Enhanced LLM Model with Classification Head

This module provides an enhanced version of the Gemma-3-270m-it model (Model A)
with additional fully connected layers for classification tasks.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel,
)
from peft import LoraConfig, get_peft_model, TaskType


class EnhancedLLMModel(nn.Module):
    """
    Enhanced LLM Model (Model A) with additional classification layers.
    
    This model combines the power of Gemma-3-270m-it with custom classification
    capabilities by adding two fully connected layers that output 12 dimensions.
    
    Attributes:
        base_model_name (str): Name of the base LLM model
        num_classes (int): Number of output classes (default: 12)
        hidden_size (int): Hidden size of the intermediate layer
        dropout_rate (float): Dropout rate for regularization
        use_lora (bool): Whether to use LoRA for efficient fine-tuning
    """
    
    def __init__(
        self,
        base_model_name: str = "google/gemma-3-270m-it",
        num_classes: int = 12,
        hidden_size: int = 256,
        dropout_rate: float = 0.1,
        use_lora: bool = True,
        lora_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the Enhanced LLM Model.
        
        Args:
            base_model_name: Name of the base model to load
            num_classes: Number of output classes for classification
            hidden_size: Size of the hidden layer in classification head
            dropout_rate: Dropout rate for regularization
            use_lora: Whether to use LoRA for parameter-efficient fine-tuning
            lora_config: Custom LoRA configuration dictionary
        """
        super().__init__()
        
        self.base_model_name = base_model_name
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.use_lora = use_lora
        
        # Load base model (Model A)
        self.base_model = AutoModel.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,  # Use float32 for MPS compatibility
            device_map="auto",
        )
        
        # Apply LoRA if specified
        if use_lora:
            self._setup_lora(lora_config)
        
        # Get the hidden size from the base model
        base_hidden_size = self.base_model.config.hidden_size
        
        # Add two fully connected layers as specified
        self.classification_head = nn.Sequential(
            # First fully connected layer
            nn.Linear(base_hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Second fully connected layer (output 12 dimensions)
            nn.Linear(hidden_size, num_classes),
        )
        
        # Initialize classification head weights
        self._init_classification_weights()
    
    def _setup_lora(self, lora_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Set up LoRA configuration for efficient fine-tuning.
        
        Args:
            lora_config: Custom LoRA configuration, if None uses default
        """
        if lora_config is None:
            lora_config = {
                "task_type": TaskType.FEATURE_EXTRACTION,
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            }
        
        config = LoraConfig(**lora_config)
        self.base_model = get_peft_model(self.base_model, config)  # type: ignore
    
    def _init_classification_weights(self) -> None:
        """Initialize the weights of the classification head."""
        for module in self.classification_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs from tokenizer
            attention_mask: Attention mask for padding tokens
            labels: Labels for classification (optional)
            **kwargs: Additional arguments for the base model
            
        Returns:
            Dictionary containing:
                - logits: Classification logits (batch_size, num_classes)
                - hidden_states: Last hidden states from base model
                - pooled_output: Pooled representation used for classification
                - loss: Classification loss (if labels provided)
        """
        # Get base model outputs
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # Get last hidden states
        hidden_states = base_outputs.last_hidden_state
        
        # Apply mean pooling across sequence length
        if attention_mask is not None:
            # Mask out padding tokens for pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
            hidden_states_masked = hidden_states * mask_expanded
            pooled_output = hidden_states_masked.sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            # Simple mean pooling without mask
            pooled_output = hidden_states.mean(dim=1)
        
        # Pass through classification head
        logits = self.classification_head(pooled_output)
        
        outputs = {
            "logits": logits,
            "hidden_states": hidden_states,
            "pooled_output": pooled_output,
        }
        
        # Calculate loss if labels are provided
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            outputs["loss"] = loss
        
        return outputs
    
    def get_tokenizer(self) -> PreTrainedTokenizer:
        """
        Get the tokenizer for this model.
        
        Returns:
            PreTrainedTokenizer: Tokenizer instance for the base model
        """
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        
        # Add pad token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenizer
    
    def freeze_base_model(self) -> None:
        """Freeze the parameters of the base model, only train classification head."""
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def unfreeze_base_model(self) -> None:
        """Unfreeze the parameters of the base model for full fine-tuning."""
        for param in self.base_model.parameters():
            param.requires_grad = True
    
    def save_model(self, save_path: str) -> None:
        """
        Save the complete model.
        
        Args:
            save_path: Path to save the model
        """
        torch.save({
            "model_state_dict": self.state_dict(),
            "model_config": {
                "base_model_name": self.base_model_name,
                "num_classes": self.num_classes,
                "hidden_size": self.hidden_size,
                "dropout_rate": self.dropout_rate,
                "use_lora": self.use_lora,
            }
        }, save_path)
    
    @classmethod
    def load_model(cls, load_path: str, device: str = "auto") -> "EnhancedLLMModel":
        """
        Load a saved model.
        
        Args:
            load_path: Path to the saved model
            device: Device to load the model on
            
        Returns:
            EnhancedLLMModel: Loaded model instance
        """
        checkpoint = torch.load(load_path, map_location="cpu")
        config = checkpoint["model_config"]
        
        model = cls(**config)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        if device != "cpu":
            model = model.to(device)
        
        return model
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "base_model": self.base_model_name,
            "num_classes": self.num_classes,
            "hidden_size": self.hidden_size,
            "dropout_rate": self.dropout_rate,
            "use_lora": self.use_lora,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_ratio": trainable_params / total_params * 100,
        }


class TextGenerationModel(nn.Module):
    """
    Text generation model wrapper for conversational tasks.
    
    This model provides an interface for text generation using the base LLM
    without the classification head.
    """
    
    def __init__(
        self,
        base_model_name: str = "google/gemma-3-270m-it",
        use_lora: bool = True,
        lora_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the text generation model.
        
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
            device_map="auto",
        )
        
        if use_lora:
            self._setup_lora(lora_config)
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _setup_lora(self, lora_config: Optional[Dict[str, Any]] = None) -> None:
        """Set up LoRA configuration for text generation."""
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
    
    def generate_text(
        self,
        prompt: str,
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
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                **kwargs
            )
        
        # Decode only the new tokens
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.shape[-1]:], 
            skip_special_tokens=True
        )
        
        return generated_text
