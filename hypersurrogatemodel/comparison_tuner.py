"""
Comparison and Tuning Module

This module provides functionality for comparing LLM outputs with dataset answers
and performing fine-tuning based on the differences.
"""

import os
# 設置環境變量以減少 Transformers 的詳細日誌輸出
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import torch
import numpy as np
import json
import psutil
import gc
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
from datasets import Dataset
from transformers import TrainingArguments, Trainer
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

from .model import TrainableLLM
from .dataset import DomainDatasetProcessor, PromptTemplate
from .trainer import TrainingMetrics, ClassificationTrainer, GenerationTrainer
from .evaluator import ModelEvaluator, PerformanceMonitor
from .utils import Logger

# Set up logger using utils.Logger
logger = Logger("comparison_tuner")


class ComparisonTuner:
    """
    Handles comparison between LLM outputs and ground truth, and performs tuning based on differences.
    """
    
    def __init__(
        self,
        model: TrainableLLM,
        tokenizer,
        output_dir: str = "./comparison_tuning_results",
        use_wandb: bool = False,
        save_files: bool = True,
    ):
        """
        Initialize the comparison tuner.
        
        Args:
            model: TrainableLLM instance
            tokenizer: Tokenizer for text processing
            output_dir: Directory to save results
            use_wandb: Whether to use Weights & Biases for logging
            save_files: Whether to save intermediate files (set False for minimal output)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.save_files = save_files
        self.output_dir = Path(output_dir)
        if self.save_files:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        
        # 設置 GPU 記憶體優化
        self._setup_gpu_memory_optimization()
        
        # 暫時禁用參數檢測以避免問題
        # self._detect_supported_parameters()
        self.supported_params = set()  # 使用空集合，只用基本參數
        
        # Initialize components
        self.dataset_processor = DomainDatasetProcessor(tokenizer)
        self.evaluator = ModelEvaluator(model, tokenizer)
        self.performance_monitor = PerformanceMonitor(str(self.output_dir / "performance_logs"))
        
    def _setup_gpu_memory_optimization(self) -> None:
        """設置 GPU 記憶體優化配置"""
        import os
        
        # 設置環境變量以避免無效參數警告
        os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # 降低日誌級別以減少警告
        
        if torch.cuda.is_available():
            # 設置記憶體分配策略
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            
            # 初始清理記憶體
            torch.cuda.empty_cache()
            
            # 記錄初始記憶體狀態
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU memory optimization enabled. Total GPU memory: {total_memory:.2f}GB")
            self._log_memory_usage("initialization")
    
    def _detect_supported_parameters(self) -> None:
        """檢測模型支持的生成參數"""
        self.supported_params = set()
        
        try:
            # 測試可能支持的生成參數（移除已知不支持的）
            test_params = {
                'use_cache': False,
                'top_k': 50,
                'top_p': 0.9,
                'repetition_penalty': 1.0,
                'length_penalty': 1.0,
            }
            
            logger.info("Detecting supported generation parameters...")
            
            for param_name, param_value in test_params.items():
                if self._test_generation_parameter(param_name, param_value):
                    self.supported_params.add(param_name)
                    logger.debug(f"✓ Parameter '{param_name}' is supported")
                else:
                    logger.debug(f"✗ Parameter '{param_name}' is not supported")
            
            logger.info(f"Supported parameters: {sorted(self.supported_params)}")
            
        except Exception as e:
            logger.warning(f"Failed to detect supported parameters: {e}. Using basic parameters only.")
            self.supported_params = set()  # 使用空集合，只用基本參數
        
    def _get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage information.
        
        Returns:
            Dictionary containing memory usage statistics
        """
        # 系統記憶體使用情況
        memory = psutil.virtual_memory()
        process = psutil.Process()
        process_memory = process.memory_info()
        
        # GPU 記憶體使用情況（如果有的話）
        gpu_memory = {}
        if torch.cuda.is_available():
            gpu_memory = {
                'gpu_allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                'gpu_reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
                'gpu_max_allocated': torch.cuda.max_memory_allocated() / 1024**3,  # GB
            }
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # MPS (Apple Silicon) 記憶體信息
            gpu_memory = {
                'mps_allocated': torch.mps.current_allocated_memory() / 1024**3 if hasattr(torch.mps, 'current_allocated_memory') else 0,
                'mps_driver_allocated': torch.mps.driver_allocated_memory() / 1024**3 if hasattr(torch.mps, 'driver_allocated_memory') else 0,
            }
        
        memory_info = {
            'system_total': memory.total / 1024**3,        # GB
            'system_available': memory.available / 1024**3, # GB
            'system_used': memory.used / 1024**3,          # GB
            'system_percent': memory.percent,               # %
            'process_rss': process_memory.rss / 1024**3,    # GB (實際使用)
            'process_vms': process_memory.vms / 1024**3,    # GB (虛擬記憶體)
        }
        
        memory_info.update(gpu_memory)
        return memory_info
        
    def _log_memory_usage(self, stage: str = "current") -> None:
        """
        Log current memory usage with stage information.
        
        Args:
            stage: Description of current processing stage
        """
        memory_info = self._get_memory_usage()
        
        logger.info(f"=== Memory Usage ({stage}) ===")
        logger.info(f"System: {memory_info['system_used']:.2f}GB/{memory_info['system_total']:.2f}GB ({memory_info['system_percent']:.1f}%)")
        logger.info(f"Process: RSS={memory_info['process_rss']:.2f}GB, VMS={memory_info['process_vms']:.2f}GB")
        
        if 'gpu_allocated' in memory_info:
            logger.info(f"GPU: Allocated={memory_info['gpu_allocated']:.2f}GB, Reserved={memory_info['gpu_reserved']:.2f}GB")
        elif 'mps_allocated' in memory_info:
            logger.info(f"MPS: Allocated={memory_info['mps_allocated']:.2f}GB, Driver={memory_info['mps_driver_allocated']:.2f}GB")
        
        logger.info("=" * 35)
        
    def _cleanup_memory(self, log_before_after: bool = False) -> None:
        """
        Perform memory cleanup operations.
        
        Args:
            log_before_after: Whether to log memory usage before and after cleanup
        """
        if log_before_after:
            memory_before = self._get_memory_usage()
            logger.debug(f"Memory before cleanup: Process={memory_before['process_rss']:.2f}GB")
        
        # Python garbage collection
        collected = gc.collect()
        
        # PyTorch cache cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Reset peak memory stats for better tracking
            torch.cuda.reset_peak_memory_stats()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
        
        if log_before_after:
            memory_after = self._get_memory_usage()
            memory_freed = memory_before['process_rss'] - memory_after['process_rss']
            logger.debug(f"Memory after cleanup: Process={memory_after['process_rss']:.2f}GB (freed {memory_freed:.2f}GB, collected {collected} objects)")
    
    def force_memory_cleanup(self) -> Dict[str, float]:
        """
        Perform aggressive memory cleanup and return memory statistics.
        
        Returns:
            Dictionary with memory statistics before and after cleanup
        """
        logger.info("Performing aggressive memory cleanup...")
        
        memory_before = self._get_memory_usage()
        
        # Multiple rounds of garbage collection
        for _ in range(3):
            gc.collect()
        
        # Clear PyTorch caches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            # Clear all caches
            if hasattr(torch.cuda, 'ipc_collect'):
                torch.cuda.ipc_collect()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
        
        # Force garbage collection again
        gc.collect()
        
        memory_after = self._get_memory_usage()
        memory_freed = memory_before['process_rss'] - memory_after['process_rss']
        
        logger.info(f"Aggressive cleanup completed: freed {memory_freed:.2f}GB")
        
        return {
            'memory_before': memory_before['process_rss'],
            'memory_after': memory_after['process_rss'],
            'memory_freed': memory_freed
        }
    
    def _check_gpu_memory_availability(self, required_gb: float = 1.0) -> bool:
        """
        檢查是否有足夠的 GPU 記憶體
        
        Args:
            required_gb: 需要的記憶體量 (GB)
            
        Returns:
            是否有足夠的記憶體
        """
        if not torch.cuda.is_available():
            return True  # CPU 模式不需要檢查 GPU 記憶體
        
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            cached_memory = torch.cuda.memory_reserved() / 1024**3
            free_memory = total_memory - cached_memory
            
            is_available = free_memory >= required_gb
            
            if not is_available:
                logger.warning(f"Insufficient GPU memory: need {required_gb:.2f}GB, have {free_memory:.2f}GB")
            
            return is_available
            
        except Exception as e:
            logger.warning(f"Failed to check GPU memory: {e}")
            return False
        
    def _auto_adjust_batch_size(self, initial_batch_size: int) -> int:
        """
        根據 GPU 記憶體自動調整批次大小
        
        Args:
            initial_batch_size: 初始批次大小
            
        Returns:
            調整後的批次大小
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, using smaller batch size")
            return min(initial_batch_size, 4)  # CPU 模式下使用較小批次
        
        try:
            # 獲取 GPU 記憶體信息
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            allocated_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            cached_memory = torch.cuda.memory_reserved() / 1024**3  # GB
            free_memory = total_memory - cached_memory
            
            # 根據可用記憶體調整批次大小
            if free_memory > 4.0:  # 超過 4GB 可用
                recommended_batch_size = min(initial_batch_size, 256)
            elif free_memory > 2.0:  # 2-4GB 可用
                recommended_batch_size = min(initial_batch_size, 128)
            elif free_memory > 1.0:  # 1-2GB 可用
                recommended_batch_size = min(initial_batch_size, 32)
            else:  # 少於 1GB 可用
                recommended_batch_size = min(initial_batch_size, 8)
            
            if recommended_batch_size != initial_batch_size:
                logger.info(f"Auto-adjusted batch size from {initial_batch_size} to {recommended_batch_size} based on available GPU memory")
            
            return recommended_batch_size
            
        except Exception as e:
            logger.warning(f"Failed to auto-adjust batch size: {e}")
            return min(initial_batch_size, 4)
    
    def _get_safe_generation_kwargs(self) -> Dict[str, Any]:
        """
        獲取安全的生成參數，避免無效參數警告
        
        Returns:
            安全的生成參數字典
        """
        # 使用最基本和絕對安全的參數，不依賴任何檢測
        safe_kwargs = {
            'max_new_tokens': 50,
            'temperature': 0.7,
            'do_sample': True,
        }
        
        return safe_kwargs
    
    def _test_generation_parameter(self, param_name: str, param_value: Any) -> bool:
        """
        測試特定生成參數是否被模型支持
        
        Args:
            param_name: 參數名稱
            param_value: 參數值
            
        Returns:
            是否支持該參數
        """
        try:
            # 對於已知不支持的參數，直接返回 False
            known_unsupported = {'early_stopping', 'cache_implementation'}
            if param_name in known_unsupported:
                return False
            
            # 使用 model.py 中的安全生成方法
            test_kwargs = {param_name: param_value}
            
            # 嘗試生成一個最小的測試輸出
            test_output = self.model.generate_text(
                prompt="test",
                max_new_tokens=1,
                do_sample=False,
                **test_kwargs
            )
            return True
            
        except Exception as e:
            logger.debug(f"Parameter {param_name} test failed: {e}")
            return False
        
    def load_and_compare_dataset(
        self,
        dataset_path: Union[str, Path],
        text_column: str = "text",
        answer_column: str = "answer",
        task_type: str = "generation",  # "classification" or "generation"
        comparison_method: str = "exact_match",  # "exact_match", "similarity", "structured"
    ) -> Dict[str, Any]:
        """
        Load dataset, generate predictions, and compare with ground truth.
        
        Args:
            dataset_path: Path to the dataset file
            text_column: Name of the text input column
            answer_column: Name of the ground truth answer column
            task_type: Type of task ("classification" or "generation")
            comparison_method: Method for comparing outputs
            
        Returns:
            Comparison results with differences and metrics
        """
        logger.setFunctionsName("load_and_compare_dataset")
        
        # Load dataset
        dataset = self.dataset_processor.load_dataset_from_file(dataset_path)
        
        logger.info(f"Loaded dataset with {len(dataset)} samples")
        logger.info(f"Columns: {dataset.column_names}")


        texts = dataset[text_column]
        ground_truth = dataset[answer_column]
        
        # Generate predictions
        predictions = self._generate_predictions(texts, task_type)
        
        
        # Compare predictions with ground truth
        comparison_results = self._compare_outputs(
            texts, predictions, ground_truth, comparison_method, task_type
        )
        
        # Save comparison results only if enabled
        if self.save_files:
            self._save_comparison_results(comparison_results)
        
        return comparison_results
    
    def _generate_predictions(
        self,
        texts: List[str],
        task_type: str,
        batch_size: int = 256  # 更保守的默認批次大小
    ) -> List[Union[str, int]]:
        """
        Generate predictions for input texts.
        
        Args:
            texts: List of input texts
            task_type: Type of task
            batch_size: Batch size for processing
            
        Returns:
            List of predictions
        """
        predictions = []
        
        logger.info(f"Generating predictions for {len(texts)} samples...")
        self._log_memory_usage("prediction generation start")
        
        # 自動調整批次大小
        adjusted_batch_size = self._auto_adjust_batch_size(batch_size)
        if adjusted_batch_size != batch_size:
            logger.info(f"Batch size adjusted from {batch_size} to {adjusted_batch_size}")
            batch_size = adjusted_batch_size
        
        # 為批次處理添加進度條
        num_batches = (len(texts) + batch_size - 1) // batch_size
        cleanup_interval = max(1, num_batches // 10)  # 每處理10%的批次清理一次記憶體
        
        with tqdm(total=len(texts), desc="Generating predictions", unit="sample") as pbar:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                if task_type == "classification":
                    batch_predictions = self._classify_batch(batch_texts)
                else:  # generation
                    batch_predictions = self._generate_batch(batch_texts)
                
                predictions.extend(batch_predictions)
                
                # 定期清理記憶體
                if batch_num % cleanup_interval == 0 or batch_num == num_batches:
                    self._cleanup_memory()
                    if batch_num % cleanup_interval == 0:
                        logger.info(f"Memory cleanup performed at batch {batch_num}/{num_batches}")
                
                pbar.update(len(batch_texts))
                memory_info = self._get_memory_usage()
                pbar.set_postfix({
                    'batch': f'{batch_num}/{num_batches}',
                    'completed': f'{len(predictions)}/{len(texts)}',
                    'memory': f'{memory_info["process_rss"]:.1f}GB'
                })
        
        self._log_memory_usage("prediction generation complete")
        return predictions
    
    def _classify_batch(self, texts: List[str]) -> List[int]:
        """Generate classification predictions for a batch of texts."""
        # Tokenize with explicit attention mask
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            return_attention_mask=True,  # 明確返回 attention mask
        )
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],  # 確保傳遞 attention mask
            )
            
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            predictions = torch.argmax(logits, dim=-1)
            
        return predictions.cpu().numpy().tolist()
    
    def _generate_batch(self, texts: List[str]) -> List[str]:
        """Generate text outputs for a batch of texts with adaptive batch sizing."""
        predictions = []
        
        # 動態調整批次大小以避免記憶體溢出
        adaptive_batch_size = self._get_adaptive_batch_size(len(texts))
        
        if adaptive_batch_size < len(texts):
            # 分割成更小的子批次
            logger.info(f"Splitting batch of {len(texts)} into sub-batches of size {adaptive_batch_size}")
            
            for i in range(0, len(texts), adaptive_batch_size):
                sub_batch = texts[i:i + adaptive_batch_size]
                sub_predictions = self._generate_sub_batch(sub_batch)
                predictions.extend(sub_predictions)
                
                # 每個子批次後清理記憶體
                self._cleanup_memory()
        else:
            # 嘗試處理整個批次
            predictions = self._generate_sub_batch(texts)
        
        return predictions
    
    def _get_adaptive_batch_size(self, requested_size: int) -> int:
        """
        根據可用 GPU 記憶體動態調整批次大小
        
        Args:
            requested_size: 請求的批次大小
            
        Returns:
            調整後的批次大小
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, using smaller batch size")
            return min(requested_size, 4)  # CPU 模式下使用較小批次
        
        try:
            # 獲取 GPU 記憶體信息
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            allocated_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            cached_memory = torch.cuda.memory_reserved() / 1024**3  # GB
            free_memory = total_memory - cached_memory
            
            # 根據可用記憶體調整批次大小
            if free_memory > 4.0:  # 超過 4GB 可用
                max_batch_size = min(requested_size, 256)
            elif free_memory > 2.0:  # 2-4GB 可用
                max_batch_size = min(requested_size, 128)
            elif free_memory > 1.0:  # 1-2GB 可用
                max_batch_size = min(requested_size, 32)
            else:  # 少於 1GB 可用
                max_batch_size = min(requested_size, 8)
            
            logger.debug(f"GPU memory: {free_memory:.2f}GB free, using batch size {max_batch_size}")
            return max_batch_size
            
        except Exception as e:
            logger.warning(f"Failed to get GPU memory info: {e}, using conservative batch size")
            return min(requested_size, 2)
    
    def _generate_sub_batch(self, texts: List[str]) -> List[str]:
        """Generate text outputs for a sub-batch of texts."""
        predictions = []
        
        try:
            # 強制清理記憶體準備生成
            self._cleanup_memory()
            
            # 真正的批次處理生成
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                return_attention_mask=True
            )
            
            # Move inputs to the same device as the model
            device = next(self.model.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                # 嘗試使用 TrainableLLM 的安全生成方法
                try:
                    generation_kwargs = self._get_safe_generation_kwargs()
                    outputs = self.model._safe_generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        **generation_kwargs
                    )
                except Exception as e:
                    logger.warning(f"Safe generation failed: {e}. Using ultra-safe method.")
                    # 回退到超級安全的方法
                    outputs = self.model._ultra_safe_generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=50
                    )
            
            # 解碼批次輸出
            for i, output in enumerate(outputs):
                # 移除輸入部分，只保留生成的新文本
                input_length = inputs["input_ids"][i].shape[0]
                generated_tokens = output[input_length:]
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                predictions.append(generated_text.strip())
            
            # 清理生成後的張量
            del outputs, inputs
            self._cleanup_memory()
                
        except torch.cuda.OutOfMemoryError as e:
            logger.warning(f"GPU OOM in sub-batch generation: {e}. Falling back to sequential generation.")
            # 如果批次生成失敗，回退到逐個生成
            predictions = self._generate_sequential(texts)
            
        except Exception as e:
            logger.warning(f"Sub-batch generation failed: {e}. Falling back to sequential generation.")
            predictions = self._generate_sequential(texts)
        
        return predictions
    
    def _generate_sequential(self, texts: List[str]) -> List[str]:
        """逐個生成文本的後備方法"""
        predictions = []
        
        # 每次生成前都清理記憶體
        self._cleanup_memory()
        
        if len(texts) > 5:
            texts_iter = tqdm(texts, desc="Generating sequentially", unit="text", leave=False)
        else:
            texts_iter = texts
            
        for i, text in enumerate(texts_iter):
            try:
                # 每 5 個文本清理一次記憶體
                if i > 0 and i % 5 == 0:
                    self._cleanup_memory()
                    
                generated = self.model.generate_text(
                    prompt=text,
                    max_new_tokens=50,  # 減少生成長度
                    temperature=0.7,
                    do_sample=True
                )
                predictions.append(generated.strip())
            except Exception as e:
                logger.warning(f"Sequential generation failed for text: {text[:50]}... Error: {e}")
                predictions.append("")
        
        return predictions
    
    def _compare_outputs(
        self,
        texts: List[str],
        predictions: List[Union[str, int]],
        ground_truth: List[Union[str, int]],
        comparison_method: str,
        task_type: str
    ) -> Dict[str, Any]:
        """
        Compare predictions with ground truth using specified method.
        
        Args:
            texts: Original input texts
            predictions: Model predictions
            ground_truth: Ground truth answers
            comparison_method: Method for comparison
            task_type: Type of task
            
        Returns:
            Comprehensive comparison results
        """
        differences = []
        matches = []
        similarity_scores = []
        MATCH_THRESHOLD = 0.8  # similarity-based matching th
        
        # 為數據比較添加進度條
        logger.info(f"Comparing {len(texts)} predictions with ground truth...")
        self._log_memory_usage("comparison start")
        
        cleanup_interval = max(100, len(texts) // 20)  # 每處理5%的樣本清理一次記憶體
        
        with tqdm(total=len(texts), desc="Comparing outputs", unit="comparison") as pbar:
            for i, (text, pred, gt) in enumerate(zip(texts, predictions, ground_truth)):
                if comparison_method == "exact_match":
                    is_match = pred == gt
                    similarity = 1.0 if is_match else 0.0
                elif comparison_method == "similarity":
                    similarity = self._calculate_similarity(pred, gt)
                    is_match = similarity > MATCH_THRESHOLD
                elif comparison_method == "structured":
                    similarity, is_match = self._structured_comparison(pred, gt)
                else:
                    raise ValueError(f"Unknown comparison method: {comparison_method}")
                
                matches.append(is_match)
                similarity_scores.append(similarity)
                
                if not is_match:
                    differences.append({
                        "index": i,
                        "input_text": text,
                        "prediction": pred,
                        "ground_truth": gt,
                        "similarity": similarity,
                        "difference_type": self._categorize_difference(pred, gt, task_type)
                    })
                
                # 定期清理記憶體
                if (i + 1) % cleanup_interval == 0:
                    self._cleanup_memory()
                    logger.debug(f"Memory cleanup performed at comparison {i+1}/{len(texts)}")
                
                # 更新進度條並顯示記憶體使用情況
                pbar.update(1)
                if (i + 1) % 50 == 0:  # 每50個比較顯示一次記憶體
                    memory_info = self._get_memory_usage()
                    pbar.set_postfix({
                        'matches': sum(matches),
                        'differences': len(differences),
                        'accuracy': f'{sum(matches)/(i+1):.3f}',
                        'memory': f'{memory_info["process_rss"]:.1f}GB'
                    })
                else:
                    pbar.set_postfix({
                        'matches': sum(matches),
                        'differences': len(differences),
                        'accuracy': f'{sum(matches)/(i+1):.3f}'
                    })
        
        self._log_memory_usage("comparison complete")
        
        # Calculate overall metrics
        accuracy = sum(matches) / len(matches)
        avg_similarity = np.mean(similarity_scores)
        
        # Error analysis
        error_analysis = self._analyze_errors(differences, task_type)
        
        # Todo; similarity 改成數字距離形式（因為輸出是數字）
        results = {
            "overall_metrics": {
                "accuracy": accuracy,
                "average_similarity": avg_similarity,
                "total_samples": len(texts),
                "correct_predictions": sum(matches),
                "incorrect_predictions": len(texts) - sum(matches),
            },
            "differences": differences,
            "error_analysis": error_analysis,
            "similarity_scores": similarity_scores,
            "comparison_method": comparison_method,
            "task_type": task_type,
            "timestamp": datetime.now().isoformat(),
        }
        
        return results
    
    def _calculate_similarity(self, pred: Union[str, int], gt: Union[str, int]) -> float:
        """Calculate similarity between prediction and ground truth."""
        if isinstance(pred, (int, float)) and isinstance(gt, (int, float)):
            return 1.0 if pred == gt else 0.0
        
        # For strings, use simple token-based similarity
        pred_str = str(pred).lower().strip()
        gt_str = str(gt).lower().strip()
        
        if pred_str == gt_str:
            return 1.0
        
        # Simple token overlap similarity
        pred_tokens = set(pred_str.split())
        gt_tokens = set(gt_str.split())
        
        if not pred_tokens and not gt_tokens:
            return 1.0
        if not pred_tokens or not gt_tokens:
            return 0.0
        
        intersection = pred_tokens & gt_tokens
        union = pred_tokens | gt_tokens
        
        return len(intersection) / len(union)
    
    def _structured_comparison(self, pred: Union[str, int], gt: Union[str, int]) -> Tuple[float, bool]:
        """Compare structured outputs (e.g., JSON)."""
        try:
            # Try to parse as JSON
            if isinstance(pred, str) and isinstance(gt, str):
                pred_json = json.loads(pred.strip())
                gt_json = json.loads(gt.strip())
                
                # Compare JSON structures
                if pred_json == gt_json:
                    return 1.0, True
                
                # Partial comparison for JSON objects
                if isinstance(pred_json, dict) and isinstance(gt_json, dict):
                    common_keys = set(pred_json.keys()) & set(gt_json.keys())
                    if common_keys:
                        matches = sum(1 for key in common_keys if pred_json[key] == gt_json[key])
                        similarity = matches / len(gt_json)
                        return similarity, similarity > 0.8
                
                return 0.0, False
            else:
                return self._calculate_similarity(pred, gt), False
        except (json.JSONDecodeError, TypeError):
            # Fall back to string similarity
            return self._calculate_similarity(pred, gt), False
    
    def _categorize_difference(self, pred: Union[str, int], gt: Union[str, int], task_type: str) -> str:
        """Categorize the type of difference between prediction and ground truth."""
        if task_type == "classification":
            return "wrong_class"
        
        # For generation tasks
        pred_str = str(pred).lower().strip()
        gt_str = str(gt).lower().strip()
        
        if not pred_str:
            return "empty_generation"
        elif len(pred_str) < len(gt_str) * 0.5:
            return "too_short"
        elif len(pred_str) > len(gt_str) * 2:
            return "too_long"
        elif self._calculate_similarity(pred, gt) > 0.5:
            return "partial_match"
        else:
            return "completely_different"
    
    def _analyze_errors(self, differences: List[Dict], task_type: str) -> Dict[str, Any]:
        """Analyze patterns in errors."""
        if not differences:
            return {"error_types": {}, "common_patterns": []}
        
        # Count error types
        error_types = defaultdict(int)
        for diff in differences:
            error_types[diff["difference_type"]] += 1
        
        # Find common patterns
        common_patterns = []
        if task_type == "generation":
            # Analyze common words in failed predictions
            failed_predictions = [str(diff["prediction"]).lower() for diff in differences]
            failed_ground_truth = [str(diff["ground_truth"]).lower() for diff in differences]
            
            # This is a simplified pattern analysis
            common_patterns.append({
                "pattern": "Length analysis",
                "avg_pred_length": np.mean([len(p.split()) for p in failed_predictions]),
                "avg_gt_length": np.mean([len(p.split()) for p in failed_ground_truth]),
            })
        
        return {
            "error_types": dict(error_types),
            "common_patterns": common_patterns,
            "most_common_error": max(error_types.items(), key=lambda x: x[1])[0] if error_types else None,
        }
    
    def _save_comparison_results(self, results: Dict[str, Any]) -> None:
        """Save comparison results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"comparison_results_{timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Comparison results saved to {results_file}")
    
    def adaptive_tuning(
        self,
        comparison_results: Dict[str, Any],
        dataset_path: Union[str, Path],
        text_column: str = "text",
        answer_column: str = "answer",
        tuning_strategy: str = "error_focused",  # "error_focused", "full_retrain", "incremental"
        max_epochs: int = 3,
        learning_rate: float = 1e-5,
    ) -> Dict[str, Any]:
        """
        Perform adaptive tuning based on comparison results.
        
        Args:
            comparison_results: Results from load_and_compare_dataset
            dataset_path: Path to the original dataset
            text_column: Name of the text input column
            answer_column: Name of the ground truth answer column
            tuning_strategy: Strategy for tuning
            max_epochs: Maximum number of training epochs
            learning_rate: Learning rate for fine-tuning
            
        Returns:
            Tuning results and updated model performance
        """
        logger.info(f"Starting adaptive tuning with strategy: {tuning_strategy}")
        self._log_memory_usage("adaptive tuning start")
        
        # Load dataset
        with tqdm(total=100, desc="Loading dataset", unit="%") as pbar:
            dataset = self.dataset_processor.load_dataset_from_file(dataset_path)
            pbar.update(30)
            
            # Prepare training data based on strategy
            pbar.set_description("Preparing training data")
            if tuning_strategy == "error_focused":
                training_data = self._prepare_error_focused_data(comparison_results, dataset, text_column, answer_column)
            elif tuning_strategy == "full_retrain":
                training_data = self._prepare_full_training_data(dataset, text_column, answer_column)
            elif tuning_strategy == "incremental":
                training_data = self._prepare_incremental_data(comparison_results, dataset, text_column, answer_column)
            else:
                raise ValueError(f"Unknown tuning strategy: {tuning_strategy}")
            pbar.update(40)
            
            if not training_data:
                logger.warning("No training data prepared. Skipping tuning.")
                return {"status": "skipped", "reason": "no_training_data"}
            
            # Convert to dataset format
            pbar.set_description("Converting to dataset format")
            train_dataset = Dataset.from_list(training_data)
            pbar.update(30)
        
        logger.info(f"Prepared {len(training_data)} training samples for {tuning_strategy} strategy")
        self._log_memory_usage("after training data preparation")
        
        # 清理準備數據後的記憶體
        self._cleanup_memory()
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / "adaptive_tuning") if self.save_files else "/tmp/temp_training",
            num_train_epochs=max_epochs,
            per_device_train_batch_size=4,
            learning_rate=learning_rate,
            warmup_steps=50,
            weight_decay=0.01,
            fp16=False,
            logging_steps=10,
            save_strategy="no" if not self.save_files else "epoch",
            save_total_limit=1 if self.save_files else 0,
            report_to="wandb" if self.use_wandb else [],
        )
        
        # Determine task type and trainer
        task_type = comparison_results.get("task_type", "classification")
        
        if task_type == "classification":
            trainer = ClassificationTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                output_dir=str(self.output_dir / "adaptive_tuning") if self.save_files else "/tmp/temp_training",
                use_wandb=self.use_wandb,
                save_files=self.save_files,
            )
        else:
            trainer = GenerationTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                output_dir=str(self.output_dir / "adaptive_tuning") if self.save_files else "/tmp/temp_training",
                use_wandb=self.use_wandb,
                save_files=self.save_files,
            )
        
        # Perform training with progress indication
        logger.info(f"Starting training for {max_epochs} epochs...")
        self._log_memory_usage("before training")
        
        with tqdm(total=max_epochs, desc="Training epochs", unit="epoch") as epoch_pbar:
            training_results = trainer.train(
                train_dataset=train_dataset,
                training_args=training_args,
            )
            epoch_pbar.update(max_epochs)  # 訓練完成後更新進度條
        
        self._log_memory_usage("after training")
        
        # 強制清理訓練後的記憶體
        cleanup_stats = self.force_memory_cleanup()
        self._log_memory_usage("after training cleanup")
        
        # Evaluate after tuning
        logger.info("Evaluating model performance after tuning...")
        post_tuning_results = self.load_and_compare_dataset(
            dataset_path=dataset_path,
            text_column=text_column,
            answer_column=answer_column,
            task_type=task_type,
            comparison_method=comparison_results["comparison_method"],
        )
        
        # Compare improvement
        improvement_analysis = self._analyze_improvement(comparison_results, post_tuning_results)
        
        # Log performance
        self.performance_monitor.log_performance(
            metrics=post_tuning_results["overall_metrics"],
            epoch=max_epochs,
            dataset_name="post_tuning",
            model_info={"tuning_strategy": tuning_strategy},
        )
        
        tuning_results = {
            "tuning_strategy": tuning_strategy,
            "training_results": training_results,
            "pre_tuning_metrics": comparison_results["overall_metrics"],
            "post_tuning_metrics": post_tuning_results["overall_metrics"],
            "improvement_analysis": improvement_analysis,
            "training_data_size": len(training_data),
            "timestamp": datetime.now().isoformat(),
        }
        
        # Save tuning results only if enabled
        if self.save_files:
            self._save_tuning_results(tuning_results)
        
        logger.success(f"Adaptive tuning completed. Accuracy improved from {comparison_results['overall_metrics']['accuracy']:.3f} to {post_tuning_results['overall_metrics']['accuracy']:.3f}")
        
        return tuning_results
    
    def _prepare_error_focused_data(
        self,
        comparison_results: Dict[str, Any],
        dataset: Dataset,
        text_column: str,
        answer_column: str,
    ) -> List[Dict[str, Any]]:
        """Prepare training data focusing on errors."""
        training_data = []
        
        # Get indices of incorrect predictions
        error_indices = [diff["index"] for diff in comparison_results["differences"]]
        task_type = comparison_results.get("task_type", "classification")
        
        logger.info(f"Preparing error-focused training data for {len(error_indices)} errors...")
        
        # 為錯誤數據準備添加進度條
        cleanup_interval = max(50, len(error_indices) // 10)  # 每處理10%的錯誤樣本清理一次記憶體
        
        with tqdm(error_indices, desc="Processing error samples", unit="sample") as pbar:
            for count, idx in enumerate(pbar, 1):
                if idx < len(dataset):
                    input_text = dataset[idx][text_column]
                    target_answer = dataset[idx][answer_column]
                    
                    if task_type == "generation":
                        # For generation tasks, format as input-output pairs
                        formatted_text = f"{input_text}\n答案: {target_answer}"
                        encoded = self.tokenizer(
                            formatted_text, 
                            truncation=True, 
                            max_length=512,
                            padding=True,
                            return_attention_mask=True,  # 明確返回 attention mask
                            return_tensors=None
                        )
                        training_data.append({
                            "input_ids": encoded["input_ids"],
                            "attention_mask": encoded["attention_mask"],  # 包含 attention mask
                            "labels": encoded["input_ids"].copy(),
                        })
                    else:
                        # For classification tasks
                        training_data.append({
                            "text": input_text,
                            "label": target_answer,
                        })
                    
                    # 定期清理記憶體
                    if count % cleanup_interval == 0:
                        self._cleanup_memory()
                        logger.debug(f"Memory cleanup performed at error sample {count}/{len(error_indices)}")
                    
                    # 更新進度條顯示
                    pbar.set_postfix({"prepared": len(training_data)})
        
        logger.info(f"Prepared {len(training_data)} error-focused training samples")
        return training_data
    
    def _prepare_full_training_data(
        self,
        dataset: Dataset,
        text_column: str,
        answer_column: str,
    ) -> List[Dict[str, Any]]:
        """Prepare full training data."""
        training_data = []
        
        logger.info(f"Preparing full training data for {len(dataset)} samples...")
        
        # 為完整數據準備添加進度條
        cleanup_interval = max(100, len(dataset) // 20)  # 每處理5%的數據清理一次記憶體
        
        with tqdm(range(len(dataset)), desc="Processing full dataset", unit="sample") as pbar:
            for i in pbar:
                input_text = dataset[i][text_column]
                target_answer = dataset[i][answer_column]
                
                # Format as input-output pairs for generation
                formatted_text = f"{input_text}\n答案: {target_answer}"
                encoded = self.tokenizer(
                    formatted_text, 
                    truncation=True, 
                    max_length=512,
                    padding=True,
                    return_attention_mask=True,  # 明確返回 attention mask
                    return_tensors=None
                )
                training_data.append({
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],  # 包含 attention mask
                    "labels": encoded["input_ids"].copy(),
                })
                
                # 定期清理記憶體
                if (i + 1) % cleanup_interval == 0:
                    self._cleanup_memory()
                    logger.debug(f"Memory cleanup performed at sample {i+1}/{len(dataset)}")
                
                # 更新進度條顯示
                pbar.set_postfix({"prepared": len(training_data)})
        
        logger.info(f"Prepared {len(training_data)} full training samples")
        return training_data
    
    def _prepare_incremental_data(
        self,
        comparison_results: Dict[str, Any],
        dataset: Dataset,
        text_column: str,
        answer_column: str,
    ) -> List[Dict[str, Any]]:
        """Prepare incremental training data (errors + some correct samples)."""
        training_data = []
        task_type = comparison_results.get("task_type", "classification")
        
        # Add all error samples
        error_indices = [diff["index"] for diff in comparison_results["differences"]]
        for idx in error_indices:
            if idx < len(dataset):
                input_text = dataset[idx][text_column]
                target_answer = dataset[idx][answer_column]
                
                if task_type == "generation":
                    formatted_text = f"{input_text}\n答案: {target_answer}"
                    encoded = self.tokenizer(
                        formatted_text, 
                        truncation=True, 
                        max_length=512,
                        return_tensors=None
                    )
                    training_data.append({
                        "input_ids": encoded["input_ids"],
                        "labels": encoded["input_ids"].copy(),
                    })
                else:
                    training_data.append({
                        "text": input_text,
                        "label": target_answer,
                    })
        
        # Add some correct samples (for stability)
        correct_indices = [i for i in range(len(dataset)) if i not in error_indices]
        num_correct_to_add = min(len(error_indices), len(correct_indices) // 2)
        
        if correct_indices and num_correct_to_add > 0:
            import random
            selected_correct = random.sample(correct_indices, num_correct_to_add)
            
            for idx in selected_correct:
                input_text = dataset[idx][text_column]
                target_answer = dataset[idx][answer_column]
                
                if task_type == "generation":
                    formatted_text = f"{input_text}\n答案: {target_answer}"
                    encoded = self.tokenizer(
                        formatted_text, 
                        truncation=True, 
                        max_length=512,
                        return_tensors=None
                    )
                    training_data.append({
                        "input_ids": encoded["input_ids"],
                        "labels": encoded["input_ids"].copy(),
                    })
                else:
                    training_data.append({
                        "text": input_text,
                        "label": target_answer,
                    })
        
        logger.info(f"Prepared {len(training_data)} incremental training samples")
        return training_data
    
    def _analyze_improvement(
        self,
        pre_results: Dict[str, Any],
        post_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze improvement after tuning."""
        pre_metrics = pre_results["overall_metrics"]
        post_metrics = post_results["overall_metrics"]
        
        improvements = {}
        for metric in ["accuracy", "average_similarity"]:
            if metric in pre_metrics and metric in post_metrics:
                improvement = post_metrics[metric] - pre_metrics[metric]
                improvement_pct = (improvement / pre_metrics[metric]) * 100 if pre_metrics[metric] > 0 else 0
                improvements[metric] = {
                    "absolute_improvement": improvement,
                    "percentage_improvement": improvement_pct,
                    "pre_value": pre_metrics[metric],
                    "post_value": post_metrics[metric],
                }
        
        return {
            "improvements": improvements,
            "overall_improvement": sum(imp["absolute_improvement"] for imp in improvements.values()) / len(improvements) if improvements else 0,
            "error_reduction": pre_metrics["incorrect_predictions"] - post_metrics["incorrect_predictions"],
        }
    
    def _save_tuning_results(self, results: Dict[str, Any]) -> None:
        """Save tuning results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"tuning_results_{timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Tuning results saved to {results_file}")