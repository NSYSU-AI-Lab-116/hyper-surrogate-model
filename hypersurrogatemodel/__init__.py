"""
Trainable Language Model Package

A comprehensive package for training and deploying language models
with LoRA fine-tuning support.
"""

from .model import TrainableLLM
from .dataset import DomainDatasetProcessor, PromptTemplate
from .trainer import (
    GenerationTrainer, 
    TrainingMetrics
)
from .evaluator import ModelEvaluator
from .utils import (
    set_random_seed,
    get_device,
    get_system_info,
    save_config,
    load_config,
    create_experiment_directory,
    ConfigManager,
    Logger
)

__version__ = "1.0.0"
__author__ = "Enhanced LLM Team"
__description__ = "Trainable Language Model with LoRA Support"

# Default configuration
DEFAULT_CONFIG = {
    "model": {
        "base_model_name": "google/gemma-3-270m-it",
        "use_lora": True,
    },
    "training": {
        "num_train_epochs": 3,
        "per_device_train_batch_size": 8,
        "learning_rate": 2e-5,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "fp16": False,  # Disabled for MPS compatibility
    },
    "dataset": {
        "max_length": 512,
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
    },
    "evaluation": {
        "batch_size": 8,
        "save_results": True,
    }
}

__all__ = [
    # Model classes
    "TrainableLLM",
    
    # Dataset processing
    "DomainDatasetProcessor",
    "PromptTemplate", 
    
    # Training
    "GenerationTrainer",
    "TrainingMetrics",
    
    # Evaluation
    "ModelEvaluator",
    
    # Utilities
    "set_random_seed",
    "get_device",
    "get_system_info",
    "save_config",
    "load_config",
    "create_experiment_directory",
    "ConfigManager",
    "Logger",
    
    # Configuration
    "DEFAULT_CONFIG",
]
