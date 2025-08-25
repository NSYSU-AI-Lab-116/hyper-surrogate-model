"""
Enhanced LLM Model Package

A comprehensive package for training and deploying enhanced language models
with classification capabilities using LoRA fine-tuning.
"""

from .model import EnhancedLLMModel, TextGenerationModel
from .dataset import DomainDatasetProcessor, PromptTemplate, DatasetAugmentor
from .trainer import (
    ClassificationTrainer, 
    GenerationTrainer, 
    TrainingManager,
    HyperparameterTuner,
    TrainingMetrics
)
from .evaluator import (
    ModelEvaluator, 
    PerformanceMonitor, 
    FeedbackCollector,
    ModelDiagnostics
)
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
__description__ = "Enhanced LLM Model with Classification Capabilities"

# Default configuration
DEFAULT_CONFIG = {
    "model": {
        "base_model_name": "google/gemma-3-270m-it",
        "num_classes": 12,
        "hidden_size": 256,
        "dropout_rate": 0.1,
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
    "EnhancedLLMModel",
    "TextGenerationModel",
    
    # Dataset processing
    "DomainDatasetProcessor",
    "PromptTemplate", 
    "DatasetAugmentor",
    
    # Training
    "ClassificationTrainer",
    "GenerationTrainer",
    "TrainingManager",
    "HyperparameterTuner",
    "TrainingMetrics",
    
    # Evaluation
    "ModelEvaluator",
    "PerformanceMonitor",
    "FeedbackCollector", 
    "ModelDiagnostics",
    
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
