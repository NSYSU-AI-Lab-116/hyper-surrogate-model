"""
Utility Functions Module

This module provides common utility functions used across the Enhanced LLM Model system.
"""

import torch
import torch.nn as nn
import numpy as np
import random
import json
import yaml
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import os
from datetime import datetime
import hashlib
import pickle
from contextlib import contextmanager
import psutil
import GPUtil

# Set up logging
logger = logging.getLogger(__name__)


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Get the best available device for training/inference.
    
    Args:
        prefer_gpu: Whether to prefer GPU over CPU
        
    Returns:
        PyTorch device
    """
    if prefer_gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using MPS device (Apple Silicon)")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device (as requested)")
    
    return device


def get_system_info() -> Dict[str, Any]:
    """
    Get comprehensive system information.
    
    Returns:
        Dictionary containing system information
    """
    info = {
        "cpu_count": psutil.cpu_count(),
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_total_gb": psutil.virtual_memory().total / (1024**3),
        "memory_available_gb": psutil.virtual_memory().available / (1024**3),
        "memory_used_percent": psutil.virtual_memory().percent,
        "disk_usage_percent": psutil.disk_usage('/').percent,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
    }
    
    # GPU information
    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name()
        info["cuda_memory_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
        info["cuda_memory_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
        
        # Additional GPU info using GPUtil
        try:
            gpus = GPUtil.getGPUs()
            info["gpu_details"] = [
                {
                    "id": gpu.id,
                    "name": gpu.name,
                    "memory_total_mb": gpu.memoryTotal,
                    "memory_used_mb": gpu.memoryUsed,
                    "memory_free_mb": gpu.memoryFree,
                    "memory_util_percent": gpu.memoryUtil * 100,
                    "gpu_util_percent": gpu.load * 100,
                    "temperature": gpu.temperature,
                }
                for gpu in gpus
            ]
        except Exception as e:
            logger.warning(f"Could not get detailed GPU info: {e}")
    
    return info


def save_config(config: Dict[str, Any], save_path: Union[str, Path]) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save the configuration
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if save_path.suffix.lower() == '.yaml':
        with open(save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    else:
        with open(save_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    logger.info(f"Configuration saved to {save_path}")


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    if config_path.suffix.lower() == '.yaml':
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    logger.info(f"Configuration loaded from {config_path}")
    return config


def create_experiment_directory(
    base_dir: str = "./experiments",
    experiment_name: Optional[str] = None,
    timestamp: bool = True,
) -> Path:
    """
    Create a directory for experiment outputs.
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Name of the experiment
        timestamp: Whether to include timestamp in directory name
        
    Returns:
        Path to the created experiment directory
    """
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    if experiment_name is None:
        experiment_name = "experiment"
    
    if timestamp:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{experiment_name}_{timestamp_str}"
    else:
        dir_name = experiment_name
    
    experiment_dir = base_path / dir_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (experiment_dir / "models").mkdir(exist_ok=True)
    (experiment_dir / "logs").mkdir(exist_ok=True)
    (experiment_dir / "results").mkdir(exist_ok=True)
    (experiment_dir / "configs").mkdir(exist_ok=True)
    
    logger.info(f"Experiment directory created: {experiment_dir}")
    return experiment_dir


def calculate_model_hash(model: nn.Module) -> str:
    """
    Calculate a hash of the model's state dict for versioning.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model hash string
    """
    model_str = str(model.state_dict())
    model_hash = hashlib.md5(model_str.encode()).hexdigest()
    return model_hash[:8]  # Return first 8 characters


def count_parameters(model: nn.Module, only_trainable: bool = False) -> int:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        only_trainable: Whether to count only trainable parameters
        
    Returns:
        Number of parameters
    """
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_model_memory_usage(model: nn.Module) -> Dict[str, float]:
    """
    Estimate model memory usage.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with memory usage information
    """
    param_size = 0
    param_sum = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    
    buffer_size = 0
    buffer_sum = 0
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    
    all_size = (param_size + buffer_size) / 1024 / 1024  # Convert to MB
    
    return {
        "parameters_mb": param_size / 1024 / 1024,
        "buffers_mb": buffer_size / 1024 / 1024,
        "total_mb": all_size,
        "parameter_count": param_sum,
        "buffer_count": buffer_sum,
    }


@contextmanager
def timer(description: str = "Operation"):
    """
    Context manager for timing operations.
    
    Args:
        description: Description of the operation being timed
    """
    start_time = datetime.now()
    logger.info(f"{description} started at {start_time.strftime('%H:%M:%S')}")
    
    try:
        yield
    finally:
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"{description} completed in {duration}")


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_save_pickle(obj: Any, path: Union[str, Path]) -> None:
    """
    Safely save object to pickle file with backup.
    
    Args:
        obj: Object to save
        path: Save path
    """
    path = Path(path)
    ensure_directory(path.parent)
    
    # Create temporary file first
    temp_path = path.with_suffix(path.suffix + '.tmp')
    
    with open(temp_path, 'wb') as f:
        pickle.dump(obj, f)
    
    # If backup exists, remove it
    backup_path = path.with_suffix(path.suffix + '.bak')
    if backup_path.exists():
        backup_path.unlink()
    
    # Move current file to backup if it exists
    if path.exists():
        path.rename(backup_path)
    
    # Move temp file to final location
    temp_path.rename(path)
    
    logger.info(f"Object saved to {path}")


def safe_load_pickle(path: Union[str, Path]) -> Any:
    """
    Safely load object from pickle file with fallback to backup.
    
    Args:
        path: Load path
        
    Returns:
        Loaded object
    """
    path = Path(path)
    
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
        
        # Try backup file
        backup_path = path.with_suffix(path.suffix + '.bak')
        if backup_path.exists():
            logger.info(f"Attempting to load backup: {backup_path}")
            with open(backup_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise e


def validate_config(config: Dict[str, Any], required_keys: List[str]) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration to validate
        required_keys: List of required keys
        
    Returns:
        True if valid, raises ValueError if not
    """
    missing_keys = [key for key in required_keys if key not in config]
    
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")
    
    return True


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override with
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def format_number(num: float, precision: int = 2) -> str:
    """
    Format large numbers with appropriate units.
    
    Args:
        num: Number to format
        precision: Decimal precision
        
    Returns:
        Formatted number string
    """
    if num >= 1e9:
        return f"{num/1e9:.{precision}f}B"
    elif num >= 1e6:
        return f"{num/1e6:.{precision}f}M"
    elif num >= 1e3:
        return f"{num/1e3:.{precision}f}K"
    else:
        return f"{num:.{precision}f}"


def calculate_eta(elapsed_time: float, current_step: int, total_steps: int) -> str:
    """
    Calculate estimated time of arrival.
    
    Args:
        elapsed_time: Time elapsed so far (seconds)
        current_step: Current step number
        total_steps: Total number of steps
        
    Returns:
        Formatted ETA string
    """
    if current_step == 0:
        return "Unknown"
    
    time_per_step = elapsed_time / current_step
    remaining_steps = total_steps - current_step
    eta_seconds = time_per_step * remaining_steps
    
    hours = int(eta_seconds // 3600)
    minutes = int((eta_seconds % 3600) // 60)
    seconds = int(eta_seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"


def create_progress_bar(
    current: int, 
    total: int, 
    length: int = 50, 
    prefix: str = "Progress"
) -> str:
    """
    Create a text-based progress bar.
    
    Args:
        current: Current progress
        total: Total items
        length: Length of progress bar
        prefix: Prefix text
        
    Returns:
        Progress bar string
    """
    if total == 0:
        return f"{prefix}: [{'='*length}] 100%"
    
    progress = current / total
    filled_length = int(length * progress)
    bar = '=' * filled_length + '-' * (length - filled_length)
    percent = progress * 100
    
    return f"{prefix}: [{bar}] {percent:.1f}% ({current}/{total})"


class ConfigManager:
    """
    Configuration manager for handling multiple configuration files.
    """
    
    def __init__(self, config_dir: str = "./configs"):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.configs = {}
    
    def load_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all configuration files from the config directory.
        
        Returns:
            Dictionary of loaded configurations
        """
        config_files = list(self.config_dir.glob("*.json")) + list(self.config_dir.glob("*.yaml"))
        
        for config_file in config_files:
            config_name = config_file.stem
            self.configs[config_name] = load_config(config_file)
        
        logger.info(f"Loaded {len(self.configs)} configuration files")
        return self.configs
    
    def get_config(self, name: str) -> Dict[str, Any]:
        """
        Get a specific configuration.
        
        Args:
            name: Configuration name
            
        Returns:
            Configuration dictionary
        """
        if name not in self.configs:
            config_path = self.config_dir / f"{name}.json"
            if not config_path.exists():
                config_path = self.config_dir / f"{name}.yaml"
            
            if config_path.exists():
                self.configs[name] = load_config(config_path)
            else:
                raise ValueError(f"Configuration '{name}' not found")
        
        return self.configs[name]
    
    def save_config(self, name: str, config: Dict[str, Any], format: str = "json") -> None:
        """
        Save a configuration.
        
        Args:
            name: Configuration name
            config: Configuration dictionary
            format: Save format ("json" or "yaml")
        """
        save_path = self.config_dir / f"{name}.{format}"
        save_config(config, save_path)
        self.configs[name] = config


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter to add colors to log messages.
    """
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def __init__(self, pattern, use_colors=True):
        super().__init__(pattern)
        self.use_colors = use_colors
    
    def format(self, record):
        if self.use_colors:
            # Add color to level name
            levelname = record.levelname
            if levelname in self.COLORS:
                colored_levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
                record.levelname = colored_levelname
        
        return super().format(record)


class Logger:
    """
    Enhanced logger with file and console output, supporting colored output.
    """
    
    def __init__(
        self, 
        name: str = "enhanced_llm", 
        log_dir: str = "./logs",
        level: str = "INFO",
        use_colors: bool = True
    ):
        """
        Initialize logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            level: Logging level
            use_colors: Enable colored console output
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.use_colors = use_colors
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        self.logger.handlers = []
        
        # Console handler with colors
        console_handler = logging.StreamHandler()
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            use_colors=use_colors
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (no colors for file output)
        log_file = self.log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
    
    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)
    
    def critical(self, message: str) -> None:
        """Log critical message."""
        self.logger.critical(message)
    
    def success(self, message: str) -> None:
        """Log success message (using info level with green color)."""
        self.info(f"✅ {message}")
    
    def step(self, message: str) -> None:
        """Log step message (using info level with special formatting)."""
        self.info(f"🔄 {message}")
    
    def result(self, message: str) -> None:
        """Log result message (using info level with special formatting)."""
        self.info(f"📊 {message}")
