"""
Configuration Manager

This module provides a centralized configuration management system
that supports YAML files, environment variables, and default values.
"""

import os
import yaml
import json
import inspect
from typing import Any, Union
from typing import get_type_hints, get_origin, get_args
from pathlib import Path
from dataclasses import dataclass
from .utils import Logger

logger = Logger("Config")


@dataclass
class ModelConfig:
    """Model configuration settings."""

    pretrained_model: str
    transfer_model_path: str
    device: str
    num_outputs: int


@dataclass
class TrainingConfig:
    """Training configuration settings."""

    train_type: str  # Options: from_pretrained / from_saved
    batch_size: int
    learning_rate: float
    num_epochs: int
    warmup_steps: int
    weight_decay: float
    fp16: bool
    gradient_accumulation_steps: int


@dataclass
class LoRAConfig:
    """LoRA configuration settings."""

    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list


@dataclass
class DatasetConfig:
    """Dataset configuration settings."""

    max_length: int
    padding: str
    truncation: bool
    template_type: str
    dataset_partition: int
    train_data_path: str
    test_data_path: str
    preprocess_train_path: str
    preprocess_test_path: str
    preprocess_source_path: str


@dataclass
class HyperConfig:
    """Path configuration settings."""

    save_basepath: str
    addition_name: str
    index_path: str | None  = None
    fs: list[dict[str,Any]] | None  = None
    new_version_dir: str | None = None

    def __post_init__(self):
        if self.save_basepath is None:
            raise ValueError("save_basepath cannot be None")
        self.index_path = os.path.join(self.save_basepath, "index.json")

        os.makedirs(self.save_basepath, exist_ok=True)
        self.index_path = os.path.join(self.save_basepath, "index.json")
        try:
            with open(self.index_path, "r") as f:
                content = f.read().strip()
                if content:
                    self.fs = json.loads(content)
                else:
                    self.fs = []
        except (json.JSONDecodeError, FileNotFoundError):
            self.fs = []

        self.fs_len = len(self.fs if self.fs is not None else [])
        if self.addition_name is not None:
            self.new_version_dir = f"v{self.fs_len}_{self.addition_name}"
        else:
            self.new_version_dir = f"v{self.fs_len + 1}"
        self.save_basepath = os.path.join(self.save_basepath, self.new_version_dir)


class ConfigManager:
    """
    Simple YAML configuration manager.
    Loads configuration from YAML file with fallback to default values.
    """

    _instance = None
    _config_loaded = False
    run_mode = None  # Options: data, train, eval

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_config(self, config_path: str | Path | None = None):
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
        self.training = self._create_training_config()
        self.lora = self._create_lora_config()
        self.dataset = self._create_dataset_config()
        self.hyper = self._create_hyper_config()

        if self.run_mode == "data":
            self._check_config_exist(
                self.dataset.preprocess_source_path,
                self.dataset.preprocess_train_path,
                self.dataset.preprocess_test_path,
                self.dataset.dataset_partition,
            )
        elif self.run_mode == "train" and self.training.train_type == "from_pretrained":
            self._check_config_exist(
                self.model.pretrained_model,
                self.training.batch_size,
                self.training.learning_rate,
                self.training.num_epochs,
                self.dataset.train_data_path,
                self.hyper.save_basepath,  # model
                self.hyper.addition_name,  # model
                self.hyper.fs,  # model
                self.hyper.new_version_dir,  # model
                self.lora.r,  # model
                self.lora.lora_alpha,  # model
                self.lora.lora_dropout,  # model
                self.lora.target_modules,  # model
            )

        elif self.run_mode == "train" and self.training.train_type == "from_saved":
            self._check_config_exist(
                self.model.transfer_model_path,
                self.training.batch_size,
                self.training.learning_rate,
                self.training.num_epochs,
                self.dataset.train_data_path,
                self.hyper.save_basepath,  # model
                self.hyper.addition_name,  # model
                self.hyper.fs,  # model
                self.hyper.new_version_dir,  # model
                self.lora.r,  # model
                self.lora.lora_alpha,  # model
                self.lora.lora_dropout,  # model
                self.lora.target_modules,  # model
            )
        elif self.run_mode == "eval":
            self._check_config_exist(
                self.model.transfer_model_path, self.dataset.test_data_path
            )

    def _load_config(self):
        """Load configuration from YAML file."""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self._config_data = yaml.safe_load(f) or {}
                print(f"✅ Loaded config from {self.config_path}")
            except Exception as e:
                print(f"❌ Failed to load config file {self.config_path}: {e}")
                self._config_data = {}
        else:
            print(f"⚠️  Config file {self.config_path} not found, using default values")
            self._config_data = {}

    def _get_config_value(
        self, section: str, key: str, default: Any = None, value_type: type = str
    ) -> Any:
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
            if yaml_value is not None:
                if not isinstance(value_type, str):
                    if isinstance(value_type,bool) and isinstance(yaml_value, str):
                        return yaml_value.lower() in ("true", "1", "yes", "on")
                    return value_type(yaml_value)
                return yaml_value if yaml_value is not None else default
        except (ValueError, TypeError):
            return default

    def _create_model_config(self) -> ModelConfig:
        """Create model configuration."""
        return ModelConfig(
            pretrained_model=self._get_config_value(
                "model", "pretrained_model", "google/gemma-2-2b-it"
            ),
            transfer_model_path=self._get_config_value(
                "model", "transfer_model_path", None
            ),
            device=self._get_config_value("model", "device", "auto"),
            num_outputs=self._get_config_value("model", "num_outputs", 1, int),
        )

    def _create_training_config(self) -> TrainingConfig:
        """Create training configuration."""
        return TrainingConfig(
            train_type=self._get_config_value(
                "training", "train_type", "from_pretrained", str
            ),
            batch_size=self._get_config_value("training", "batch_size", 8, int),
            learning_rate=self._get_config_value(
                "training", "learning_rate", 2e-5, float
            ),
            num_epochs=self._get_config_value("training", "num_epochs", 3, int),
            warmup_steps=self._get_config_value("training", "warmup_steps", 100, int),
            weight_decay=self._get_config_value(
                "training", "weight_decay", 0.01, float
            ),
            fp16=self._get_config_value("training", "fp16", False, bool),
            gradient_accumulation_steps=self._get_config_value(
                "training", "gradient_accumulation_steps", 1, int
            ),
        )

    def _create_lora_config(self) -> LoRAConfig:
        """Create LoRA configuration."""
        default_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        return LoRAConfig(
            r=self._get_config_value("lora", "r", 16, int),
            lora_alpha=self._get_config_value("lora", "lora_alpha", 32, int),
            lora_dropout=self._get_config_value("lora", "lora_dropout", 0.1, float),
            target_modules=self._get_config_value(
                "lora", "target_modules", default_target_modules, list
            ),
        )

    def _create_dataset_config(self) -> DatasetConfig:
        """Create dataset configuration."""
        return DatasetConfig(
            max_length=self._get_config_value("dataset", "max_length", 512, int),
            padding=self._get_config_value("dataset", "padding", True, bool),
            truncation=self._get_config_value("dataset", "truncation", True, bool),
            template_type=self._get_config_value(
                "dataset", "template_type", "structured"
            ),
            dataset_partition=self._get_config_value(
                "dataset", "dataset_partition", -1, int
            ),
            train_data_path=self._get_config_value(
                "dataset", "train_data_path", None, str
            ),
            test_data_path=self._get_config_value(
                "dataset", "test_data_path", None, str
            ),
            preprocess_train_path=self._get_config_value(
                "dataset", "preprocess_train_path", None, str
            ),
            preprocess_test_path=self._get_config_value(
                "dataset", "preprocess_test_path", None, str
            ),
            preprocess_source_path=self._get_config_value(
                "dataset", "preprocess_source_path", None, str
            ),
        )

    def _create_hyper_config(self) -> HyperConfig:
        """Create path configuration."""
        return HyperConfig(
            save_basepath=self._get_config_value(
                "paths", "save_basepath", "./saved_model", str
            ),
            addition_name=self._get_config_value("paths", "addition_name", None),
        )

    def _check_config_exist(self, *args):
        """
        check var is exist and in valid type
        """
        quitting = False

        frame = inspect.currentframe().f_back  # type: ignore

        for i, arg_value in enumerate(args):
            var_path = self._trace_variable_path(frame, i)
            expected_type = self._get_expected_type(var_path)
            if not self._is_valid_type(arg_value, expected_type):
                actual_type = type(arg_value).__name__
                logger.error(
                    f"Configuration error: Invalid type for {var_path}. Expected: {expected_type}, Got: {actual_type} ({arg_value})"
                )
                quitting = True

        if quitting:
            raise ValueError("Configuration values have invalid types")

    def _get_expected_type(self, var_path: str):
        """
        'model.pretrained_model' -> str|None
        """
        try:
            parts = var_path.split(".")
            if len(parts) >= 2:
                section = parts[0]  # model, training, dataset, etc.
                field_name = parts[1]  # pretrained_model, batch_size, etc.

                config_class = None
                if section == "model":
                    config_class = ModelConfig
                elif section == "training":
                    config_class = TrainingConfig
                elif section == "dataset":
                    config_class = DatasetConfig
                elif section == "lora":
                    config_class = LoRAConfig
                elif section == "hyper":
                    config_class = HyperConfig

                if config_class:
                    type_hints = get_type_hints(config_class)
                    return type_hints.get(field_name, "Any")

            return "Any"
        except Exception:
            return "Any"

    def _is_valid_type(self, value, expected_type) -> bool:
        if expected_type == "Any":
            return True

        if value is None:
            if hasattr(expected_type, "__origin__"):
                if get_origin(expected_type) is Union:
                    return type(None) in get_args(expected_type)
            return 'None' in str(expected_type)
        
        if hasattr(expected_type, '__origin__'):
            if get_origin(expected_type) is Union:
                valid_types = [
                    t for t in get_args(expected_type) if t is not type(None)
                ]
                return any(isinstance(value, t) for t in valid_types)

        return isinstance(value, expected_type)

    def _trace_variable_path(self, frame, arg_index: int) -> str:
        try:
            filename = frame.f_code.co_filename
            line_number = frame.f_lineno

            with open(filename, "r", encoding="utf-8") as f:
                lines = f.readlines()

            call_lines = []
            start_line = line_number - 1

            for i in range(start_line, max(0, start_line - 5), -1):
                if "check_config_exist" in lines[i]:
                    start_line = i
                    break

            paren_count = 0
            for i in range(start_line, min(len(lines), start_line + 20)):
                line = lines[i].strip()
                call_lines.append(line)

                paren_count += line.count("(") - line.count(")")

                if paren_count == 0 and "check_config_exist" in "".join(call_lines):
                    break

            call_text = " ".join(call_lines)

            import re

            match = re.search(
                r"check_config_exist\s*\(\s*(.*?)\s*\)", call_text, re.DOTALL
            )
            if match:
                args_str = match.group(1)
                args_list = self._split_arguments(args_str)

                if arg_index < len(args_list):
                    arg_expr = args_list[arg_index].strip()
                    if arg_expr.startswith("self."):
                        return arg_expr[5:]
                    return arg_expr

            return f"argument_{arg_index}"

        except Exception:
            return f"argument_{arg_index}"

    def _split_arguments(self, args_str: str) -> list:
        args = []
        current_arg = ""
        paren_depth = 0
        bracket_depth = 0

        for char in args_str:
            if char in "([":
                paren_depth += 1 if char == "(" else 0
                bracket_depth += 1 if char == "[" else 0
                current_arg += char
            elif char in ")]":
                paren_depth -= 1 if char == ")" else 0
                bracket_depth -= 1 if char == "]" else 0
                current_arg += char
            elif char == "," and paren_depth == 0 and bracket_depth == 0:
                if current_arg.strip():
                    args.append(current_arg.strip())
                current_arg = ""
            else:
                current_arg += char
        if current_arg.strip():
            args.append(current_arg.strip())

        return args

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""

        return {
            "model": {
                "pretrained_model": self.model.pretrained_model,
                "transfer_model_path": self.model.transfer_model_path,
                "device": self.model.device,
                "num_outputs": self.model.num_outputs,
            },
            "training": {
                "train_type": self.training.train_type,
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
            "dataset": {
                "max_length": self.dataset.max_length,
                "padding": self.dataset.padding,
                "truncation": self.dataset.truncation,
                "template_type": self.dataset.template_type,
                "dataset_partition": self.dataset.dataset_partition,
                "train_data_path": self.dataset.train_data_path,
                "test_data_path": self.dataset.test_data_path,
                "preprocess_train_path": self.dataset.preprocess_train_path,
                "preprocess_test_path": self.dataset.preprocess_test_path,
                "preprocess_srouce_path": self.dataset.preprocess_source_path,
            },
            "hyper": {
                "save_basepath": self.hyper.save_basepath,
                "addition_name": self.hyper.addition_name,
                "index_path": self.hyper.index_path,
                "fs": self.hyper.fs,
            },
        }
    
    def save_config(self, path: str | Path | None = None):
        """Save current configuration to YAML file."""
        save_path = Path(path) if path else self.config_path
        with open(save_path, "w", encoding="utf-8") as f:
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


def get_config(config_path: str | Path | None = None):
    """Get the singleton configuration instance."""
    config_manager = ConfigManager()
    if not config_manager._config_loaded:
        config_manager.load_config(config_path)
    return config_manager


class LazyConfig:
    def __getattr__(self, name):
        config_instance = get_config()
        for attr_name in dir(config_instance):
            if not attr_name.startswith("_"):
                setattr(self, attr_name, getattr(config_instance, attr_name))
        return getattr(self, name)


config = LazyConfig()
