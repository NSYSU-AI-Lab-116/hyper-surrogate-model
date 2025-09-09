"""
Configuration Manager

This module provides a centralized configuration management system
that supports YAML files, environment variables, and default values.
"""

import os
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Model configuration settings."""
    pretrained_model: str = "google/gemma-3-270m-it"
    use_lora: bool = True
    device: str = "auto"


@dataclass
class GenerationConfig:
    """Generation configuration settings."""
    max_new_tokens: int = 50
    temperature: float = 0.7
    do_sample: bool = True
    top_k: int = 64
    top_p: float = 0.95
    repetition_penalty: float = 1.1
    length_penalty: float = 1.0
    num_beams: int = 1


@dataclass
class TrainingConfig:
    """Training configuration settings."""
    batch_size: int = 8
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    fp16: bool = False
    gradient_accumulation_steps: int = 1


@dataclass
class LoRAConfig:
    """LoRA configuration settings."""
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: str = "INFO"
    use_wandb: bool = False
    wandb_project: str = "hypersurrogatemodel"
    save_files: bool = True
    output_dir: str = "./results"


@dataclass
class DatasetConfig:
    """Dataset configuration settings."""
    max_length: int = 512
    padding: str = "max_length"
    truncation: bool = True
    template_type: str = "generation"


@dataclass
class ComparisonConfig:
    """Comparison and tuning configuration settings."""
    method: str = "similarity"
    batch_size: int = 256
    similarity_threshold: float = 0.8
    tuning_strategy: str = "error_focused"


class ConfigManager:
    """
    Simple YAML configuration manager.
    
    Loads configuration from YAML file with fallback to default values.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to YAML configuration file (default: config.yaml)
        """
        if config_path:
            self.config_path = Path(config_path)
        else:
            self.config_path = Path("config.yaml")
            
        self._config_data = {}
        self._load_config()
        
        # Initialize configuration objects
        self.model = self._create_model_config()
        self.generation = self._create_generation_config()
        self.training = self._create_training_config()
        self.lora = self._create_lora_config()
        self.logging = self._create_logging_config()
        self.dataset = self._create_dataset_config()
        self.comparison = self._create_comparison_config()
    
    def _load_config(self):
        """Load configuration from YAML file."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self._config_data = yaml.safe_load(f) or {}
                print(f"✅ Loaded config from {self.config_path}")
            except Exception as e:
                print(f"❌ Failed to load config file {self.config_path}: {e}")
                self._config_data = {}
        else:
            print(f"⚠️  Config file {self.config_path} not found, using default values")
            self._config_data = {}
    
    def _get_config_value(self, section: str, key: str, default: Any = None, value_type: type = str) -> Any:
        """
        Get configuration value from YAML with fallback to default.
        
        Args:
            section: Configuration section name
            key: Configuration key name
            default: Default value
            value_type: Type to convert to
            
        Returns:
            Configuration value
        """
        try:
            yaml_value = self._config_data.get(section, {}).get(key, default)
            if yaml_value is not None and value_type != str:
                if value_type == bool and isinstance(yaml_value, str):
                    return yaml_value.lower() in ('true', '1', 'yes', 'on')
                return value_type(yaml_value)
            return yaml_value if yaml_value is not None else default
        except (ValueError, TypeError):
            return default
    
    def _create_model_config(self) -> ModelConfig:
        """Create model configuration."""
        return ModelConfig(
            pretrained_model=self._get_config_value("model", "pretrained_model", "google/gemma-2-2b-it"),
            use_lora=self._get_config_value("model", "use_lora", True, bool),
            device=self._get_config_value("model", "device", "auto"),
        )

    def _create_generation_config(self) -> GenerationConfig:
        """Create generation configuration."""
        return GenerationConfig(
            max_new_tokens=self._get_config_value("generation", "max_new_tokens", 50, int),
            temperature=self._get_config_value("generation", "temperature", 0.7, float),
            do_sample=self._get_config_value("generation", "do_sample", True, bool),
            top_k=self._get_config_value("generation", "top_k", 64, int),
            top_p=self._get_config_value("generation", "top_p", 0.95, float),
            repetition_penalty=self._get_config_value("generation", "repetition_penalty", 1.1, float),
            length_penalty=self._get_config_value("generation", "length_penalty", 1.0, float),
            num_beams=self._get_config_value("generation", "num_beams", 1, int),
        )

    def _create_training_config(self) -> TrainingConfig:
        """Create training configuration."""
        return TrainingConfig(
            batch_size=self._get_config_value("training", "batch_size", 8, int),
            learning_rate=self._get_config_value("training", "learning_rate", 2e-5, float),
            num_epochs=self._get_config_value("training", "num_epochs", 3, int),
            warmup_steps=self._get_config_value("training", "warmup_steps", 100, int),
            weight_decay=self._get_config_value("training", "weight_decay", 0.01, float),
            fp16=self._get_config_value("training", "fp16", False, bool),
            gradient_accumulation_steps=self._get_config_value("training", "gradient_accumulation_steps", 1, int),
        )

    def _create_lora_config(self) -> LoRAConfig:
        """Create LoRA configuration."""
        default_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        return LoRAConfig(
            r=self._get_config_value("lora", "r", 16, int),
            lora_alpha=self._get_config_value("lora", "lora_alpha", 32, int),
            lora_dropout=self._get_config_value("lora", "lora_dropout", 0.1, float),
            target_modules=self._get_config_value("lora", "target_modules", default_target_modules, list),
        )
    
    def _create_logging_config(self) -> LoggingConfig:
        """Create logging configuration."""
        return LoggingConfig(
            level=self._get_config_value("logging", "level", "INFO"),
            use_wandb=self._get_config_value("logging", "use_wandb", False, bool),
            wandb_project=self._get_config_value("logging", "wandb_project", "hypersurrogatemodel"),
            save_files=self._get_config_value("logging", "save_files", True, bool),
            output_dir=self._get_config_value("logging", "output_dir", "./results"),
        )
    
    def _create_dataset_config(self) -> DatasetConfig:
        """Create dataset configuration."""
        return DatasetConfig(
            max_length=self._get_config_value("dataset", "max_length", 512, int),
            padding=self._get_config_value("dataset", "padding", True, bool),
            truncation=self._get_config_value("dataset", "truncation", True, bool),
            template_type=self._get_config_value("dataset", "template_type", "structured"),
        )
    
    def _create_comparison_config(self) -> ComparisonConfig:
        """Create comparison configuration."""
        return ComparisonConfig(
            method=self._get_config_value("comparison", "method", "similarity"),
            batch_size=self._get_config_value("comparison", "batch_size", 32, int),
            similarity_threshold=self._get_config_value("comparison", "similarity_threshold", 0.8, float),
            tuning_strategy=self._get_config_value("comparison", "tuning_strategy", "error_focused"),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model": {
                "pretrained_model": self.model.pretrained_model,
                "use_lora": self.model.use_lora,
                "device": self.model.device,
            },
            "generation": {
                "max_new_tokens": self.generation.max_new_tokens,
                "temperature": self.generation.temperature,
                "do_sample": self.generation.do_sample,
                "top_k": self.generation.top_k,
                "top_p": self.generation.top_p,
                "repetition_penalty": self.generation.repetition_penalty,
                "length_penalty": self.generation.length_penalty,
                "num_beams": self.generation.num_beams,
            },
            "training": {
                "batch_size": self.training.batch_size,
                "learning_rate": self.training.learning_rate,
                "num_epochs": self.training.num_epochs,
                "warmup_steps": self.training.warmup_steps,
                "weight_decay": self.training.weight_decay,
                "fp16": self.training.fp16,
                "gradient_accumulation_steps": self.training.gradient_accumulation_steps,
            },
            "lora": {
                "r": self.lora.r,
                "lora_alpha": self.lora.lora_alpha,
                "lora_dropout": self.lora.lora_dropout,
                "target_modules": self.lora.target_modules,
            },
            "logging": {
                "level": self.logging.level,
                "use_wandb": self.logging.use_wandb,
                "wandb_project": self.logging.wandb_project,
                "save_files": self.logging.save_files,
                "output_dir": self.logging.output_dir,
            },
            "dataset": {
                "max_length": self.dataset.max_length,
                "padding": self.dataset.padding,
                "truncation": self.dataset.truncation,
                "template_type": self.dataset.template_type,
            },
            "comparison": {
                "method": self.comparison.method,
                "batch_size": self.comparison.batch_size,
                "similarity_threshold": self.comparison.similarity_threshold,
                "tuning_strategy": self.comparison.tuning_strategy,
            },
        }
    
    def save_config(self, path: Optional[Union[str, Path]] = None):
        """Save current configuration to YAML file."""
        save_path = Path(path) if path else self.config_path
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def print_config(self):
        """Print current configuration."""
        print("Current Configuration:")
        print("=" * 50)
        config_dict = self.to_dict()
        for section, values in config_dict.items():
            print(f"\n[{section.upper()}]")
            for key, value in values.items():
                print(f"  {key}: {value}")


# Global configuration instance
config = ConfigManager()
