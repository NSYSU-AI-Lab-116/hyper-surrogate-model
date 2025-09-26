"""
Utility Functions Module

This module provides common utility functions used across the Enhanced LLM Model system.
"""

import torch
import numpy as np
import random
import logging
import os 
from typing import Dict, Any, List, Optional, Union
import GPUtil
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
        level: str = "INFO",
        use_colors: bool = True,
    ):
        """
        Initialize logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            level: Logging level
            use_colors: Enable colored console output
        """
        self.use_colors = use_colors
        self.function = "main"
        os.makedirs("log", exist_ok=True)
        self.logfile = "log/.log"
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
        
        open(self.logfile, 'a').close()
        file_handler = logging.FileHandler(self.logfile)
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


def get_system_info() -> dict[str, Any]:
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

def get_gpu_utilization() -> Dict[str, Any]:
    """
    Get GPU utilization percentage and memory usage using GPUtil.
    
    Returns:
        Dictionary containing GPU utilization information
    """
    gpu_info = {
        "available": False,
        "devices": [],
        "total_memory_gb": 0.0,
        "used_memory_gb": 0.0,
        "free_memory_gb": 0.0,
        "memory_utilization_percent": 0.0,
        "gpu_utilization_percent": 0.0
    }
    
    if not torch.cuda.is_available():
        logger.debug("CUDA not available")
        return gpu_info
    
    gpu_info["available"] = True
    try:
        # Get all GPUs
        gpus = GPUtil.getGPUs()
        
        total_gpu_util = 0.0
        total_memory_used = 0.0
        total_memory_total = 0.0
        for gpu in gpus:  # 修復：直接迭代 gpus，不是 gpus[0]
            # GPUtil provides memory in MB, convert to GB
            memory_used_gb = gpu.memoryUsed / 1024
            memory_total_gb = gpu.memoryTotal / 1024
            memory_free_gb = gpu.memoryFree / 1024
            memory_util_percent = gpu.memoryUtil * 100  # GPUtil gives as fraction

            device_info = {
                "id": gpu.id,
                "name": gpu.name,
                "gpu_utilization_percent": gpu.load * 100,  # GPUtil gives as fraction
                "memory_used_gb": memory_used_gb,
                "memory_total_gb": memory_total_gb,
                "memory_free_gb": memory_free_gb,
                "memory_utilization_percent": memory_util_percent,
                "temperature_c": gpu.temperature,
            }
            
            gpu_info["devices"].append(device_info)
            total_gpu_util += gpu.load * 100
            total_memory_used += memory_used_gb
            total_memory_total += memory_total_gb
        
        # Calculate averages
        gpu_count = len(gpus)
        gpu_info["gpu_utilization_percent"] = total_gpu_util / gpu_count if gpu_count > 0 else 0.0
        gpu_info["total_memory_gb"] = total_memory_total
        gpu_info["used_memory_gb"] = total_memory_used
        gpu_info["free_memory_gb"] = total_memory_total - total_memory_used
        gpu_info["memory_utilization_percent"] = (total_memory_used / total_memory_total * 100) if total_memory_total > 0 else 0.0
        
    except Exception as e:
        logger.warning(f"Failed to get GPU utilization with GPUtil: {e}")
    
    return gpu_info