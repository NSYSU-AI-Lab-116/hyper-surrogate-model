"""
Comparison and Tuning Module

This module provides functionality for comparing LLM outputs with dataset answers
and performing fine-tuning based on the differences.
"""

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
        
        # Initialize components
        self.dataset_processor = DomainDatasetProcessor(tokenizer)
        self.evaluator = ModelEvaluator(model, tokenizer)
        self.performance_monitor = PerformanceMonitor(str(self.output_dir / "performance_logs"))
        
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
        
    def _cleanup_memory(self) -> None:
        """
        Perform memory cleanup operations.
        """
        # Python garbage collection
        gc.collect()
        
        # PyTorch cache cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
        
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
        
        # 初始記憶體使用情況
        self._log_memory_usage("dataset loading start")
        
        # Load dataset
        dataset = self.dataset_processor.load_dataset_from_file(dataset_path)
        
        logger.info(f"Loaded dataset with {len(dataset)} samples")
        logger.info(f"Columns: {dataset.column_names}")
        
        # 載入數據後的記憶體使用情況
        self._log_memory_usage("after dataset loading")
        
        # Extract texts and ground truth answers
        texts = dataset[text_column]
        ground_truth = dataset[answer_column]
        
        # Generate predictions
        predictions = self._generate_predictions(texts, task_type)
        
        # 生成預測後的記憶體使用情況
        self._log_memory_usage("after predictions generation")
        
        # Compare predictions with ground truth
        comparison_results = self._compare_outputs(
            texts, predictions, ground_truth, comparison_method, task_type
        )
        
        # 比較完成後的記憶體使用情況
        self._log_memory_usage("after comparison")
        
        # 清理記憶體
        self._cleanup_memory()
        self._log_memory_usage("after memory cleanup")
        
        # Save comparison results only if enabled
        if self.save_files:
            self._save_comparison_results(comparison_results)
        
        return comparison_results
    
    def _generate_predictions(
        self,
        texts: List[str],
        task_type: str,
        batch_size: int = 8
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
        
        # 為批次處理添加進度條
        num_batches = (len(texts) + batch_size - 1) // batch_size
        with tqdm(total=len(texts), desc="Generating predictions", unit="sample") as pbar:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                if task_type == "classification":
                    batch_predictions = self._classify_batch(batch_texts)
                else:  # generation
                    batch_predictions = self._generate_batch(batch_texts)
                
                predictions.extend(batch_predictions)
                
                # 更新進度條
                pbar.update(len(batch_texts))
                
                # 獲取當前記憶體使用情況並顯示在進度條中
                memory_info = self._get_memory_usage()
                pbar.set_postfix({
                    'batch': f'{i//batch_size + 1}/{num_batches}',
                    'completed': f'{len(predictions)}/{len(texts)}',
                    'memory': f'{memory_info["process_rss"]:.1f}GB'
                })
                
                # 每10個批次清理一次記憶體
                if (i // batch_size + 1) % 10 == 0:
                    self._cleanup_memory()
        
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
        """Generate text outputs for a batch of texts."""
        predictions = []
        
        # 如果批次大小較大（超過5個），則顯示內部進度
        if len(texts) > 5:
            texts_iter = tqdm(texts, desc="Generating batch", unit="text", leave=False)
        else:
            texts_iter = texts
            
        for text in texts_iter:
            try:
                generated = self.model.generate_text(
                    prompt=text,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True
                )
                predictions.append(generated.strip())
            except Exception as e:
                logger.warning(f"Generation failed for text: {text[:50]}... Error: {e}")
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
        
        # 清理訓練後的記憶體
        self._cleanup_memory()
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
        with tqdm(error_indices, desc="Processing error samples", unit="sample") as pbar:
            for idx in pbar:
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