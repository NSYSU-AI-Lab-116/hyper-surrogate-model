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
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import os
from datetime import datetime

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
        self.function = "main"
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        self.logger.propagate = False
        self.logger.handlers = []
        
        console_handler = logging.StreamHandler()
        console_formatter = ColoredFormatter(
            '%(levelname)s - %(name)s - %(function)s - %(message)s',
            use_colors=use_colors
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        log_file = self.log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(function)s - %(filename)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
    
    def _log_with_function(self, level, message):
        """Internal method to add function name to log record."""
        record = self.logger.makeRecord(
            self.logger.name, level, __file__, 0, message, (), None
        )
        record.function = self.function
        if self.logger.isEnabledFor(level):
            self.logger.handle(record)
    
    def info(self, message: str) -> None:
        """Log info message."""
        self._log_with_function(logging.INFO, message)
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self._log_with_function(logging.WARNING, message)
    
    def error(self, message: str) -> None:
        """Log error message."""
        self._log_with_function(logging.ERROR, message)
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        self._log_with_function(logging.DEBUG, message)
    
    def critical(self, message: str) -> None:
        """Log critical message."""
        self._log_with_function(logging.CRITICAL, message)
    
    def success(self, message: str) -> None:
        """Log success message (using info level with green color)."""
        self._log_with_function(logging.INFO, f"✅ {message}")
    
    def step(self, message: str) -> None:
        """Log step message (using info level with special formatting)."""
        self._log_with_function(logging.INFO, f"🔄 {message}")
    
    def result(self, message: str) -> None:
        """Log result message (using info level with special formatting)."""
        self._log_with_function(logging.INFO, f"📊 {message}")
        
    def setFunctionsName(self, functionName = 'main') -> None:
        """Set the current function name for logging context."""
        self.function = functionName
    
logger = Logger("utils")

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
    
    logger.debug(f"Random seed set to {seed}")


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
            device = torch.device("cuda:0")
            logger.debug(f"Using CUDA device: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.debug("Using MPS device (Apple Silicon)")
        else:
            device = torch.device("cpu")
            logger.debug("Using CPU device")
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



