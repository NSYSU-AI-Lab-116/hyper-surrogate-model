"""
Training Interface Module

This module provides a comprehensive training interface for the Enhanced LLM Model
with support for both classification and generation tasks.
"""

import torch
from typing import Dict, Any, Optional, List, Callable, Union
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from datasets import Dataset, DatasetDict
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from pathlib import Path
import json
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader

from .model import TrainableLLM, TextGenerationModel
from .dataset import DomainDatasetProcessor
from .utils import Logger

# Set up logger using utils.Logger
logger = Logger("trainer")


class TrainingMetrics:
    """
    Utility class for computing and tracking training metrics.
    """
    
    @staticmethod
    def compute_classification_metrics(eval_pred) -> Dict[str, float]:
        """
        Compute classification metrics.
        
        Args:
            eval_pred: Tuple of (predictions, labels)
            
        Returns:
            Dictionary of computed metrics
        """
        predictions, labels = eval_pred
        
        # Handle case where predictions is a tuple (logits, ...)
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        # Ensure predictions is numpy array
        if not isinstance(predictions, np.ndarray):
            predictions = np.array(predictions)
        
        # Get predicted class (argmax of logits)
        if predictions.ndim > 1:
            predictions = np.argmax(predictions, axis=1)
        
        # Compute metrics
        accuracy = accuracy_score(labels, predictions)
        
        # Suppress warnings for zero division in precision/recall
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, predictions, average='weighted', zero_division=0
            )
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
        }
    
    @staticmethod
    def compute_generation_metrics(eval_pred) -> Dict[str, float]:
        """
        Compute text generation metrics.
        
        Args:
            eval_pred: Tuple of (predictions, labels)
            
        Returns:
            Dictionary of computed metrics
        """
        predictions, labels = eval_pred
        
        # For generation tasks, we typically compute perplexity
        # This is a simplified implementation
        losses = np.array(predictions) if isinstance(predictions, list) else predictions
        perplexity = np.exp(np.mean(losses))
        
        return {
            'perplexity': perplexity,
        }


class ClassificationTrainer:
    """
    Trainer class for classification tasks using the Enhanced LLM Model.
    """
    
    def __init__(
        self,
        model: TrainableLLM,
        tokenizer,
        output_dir: str = "./results",
        use_wandb: bool = False,
        wandb_project: str = "enhanced-llm-classification",
        save_files: bool = True,
    ):
        """
        Initialize the classification trainer.
        
        Args:
            model: Enhanced LLM model instance
            tokenizer: Tokenizer for text processing
            output_dir: Directory to save training outputs
            use_wandb: Whether to use Weights & Biases for logging
            wandb_project: W&B project name
        """
        self.model = model
        self.tokenizer = tokenizer
        self.save_files = save_files
        self.output_dir = Path(output_dir)
        if self.save_files:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project=wandb_project)
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        training_args: Optional[TrainingArguments] = None,
        data_collator: Optional[Callable] = None,
        callbacks: Optional[List] = None,
    ) -> Dict[str, Any]:
        """
        Train the classification model.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            training_args: Training arguments
            data_collator: Data collator for batching
            callbacks: Training callbacks
            
        Returns:
            Training results and metrics
        """
        # Set default training arguments
        if training_args is None:
            training_args = TrainingArguments(
                output_dir=str(self.output_dir),
                num_train_epochs=3,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                warmup_steps=100,
                weight_decay=0.01,
                learning_rate=2e-5,
                fp16=False,  # Disable for MPS compatibility
                logging_dir=str(self.output_dir / "logs"),
                logging_steps=10,
                eval_strategy="steps" if eval_dataset else "no",
                eval_steps=100 if eval_dataset else None,
                save_strategy="steps",
                save_steps=500,
                save_total_limit=3,
                load_best_model_at_end=True if eval_dataset else False,
                metric_for_best_model="eval_f1" if eval_dataset else None,
                greater_is_better=True,
                report_to="wandb" if self.use_wandb else [],
                dataloader_pin_memory=False,  # For MPS compatibility
            )
        
        # Set default data collator
        if data_collator is None:
            data_collator = DataCollatorWithPadding(
                tokenizer=self.tokenizer,
                padding=True,
                max_length=512,
            )
        
        # Set default callbacks
        if callbacks is None:
            callbacks = []
            if eval_dataset:
                callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=TrainingMetrics.compute_classification_metrics,
            callbacks=callbacks,
        )
        
        # Start training
        logger.info("Starting classification training...")
        train_result = trainer.train()
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Evaluate if eval dataset is provided
        eval_results = {}
        if eval_dataset:
            logger.info("Evaluating model...")
            eval_results = trainer.evaluate()
        
        # Save training results
        results = {
            "train_results": train_result,
            "eval_results": eval_results,
            "model_info": self.model.get_model_info(),
        }
        
        with open(self.output_dir / "training_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Training completed. Results saved to {self.output_dir}")
        return results
    
    def evaluate_model(
        self,
        test_dataset: Dataset,
        batch_size: int = 8,
    ) -> Dict[str, Any]:
        """
        Evaluate the trained model on test dataset.
        
        Args:
            test_dataset: Test dataset
            batch_size: Batch size for evaluation
            
        Returns:
            Evaluation results
        """
        self.model.eval()
        
        predictions = []
        true_labels = []
        
        # Create data loader
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True,
        )
        
        dataloader = DataLoader(
            test_dataset,  # type: ignore
            batch_size=batch_size,
            collate_fn=data_collator,
        )
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Move to device - get device from model parameters
                device = next(self.model.parameters()).device
                batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                
                # Get predictions
                logits = outputs["logits"]
                batch_predictions = torch.argmax(logits, dim=-1).cpu().numpy()
                batch_labels = batch["labels"].cpu().numpy()
                
                predictions.extend(batch_predictions)
                true_labels.extend(batch_labels)
        
        # Compute metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        
        # Generate classification report
        report = classification_report(
            true_labels, predictions, output_dict=True
        )
        
        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "classification_report": report,
        }
        
        logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        return results


class GenerationTrainer:
    """
    Trainer class for text generation tasks.
    """
    
    def __init__(
        self,
        model: TextGenerationModel,
        tokenizer,
        output_dir: str = "./results",
        use_wandb: bool = False,
        wandb_project: str = "enhanced-llm-generation",
        save_files: bool = True,
    ):
        """
        Initialize the generation trainer.
        
        Args:
            model: Text generation model instance
            tokenizer: Tokenizer for text processing
            output_dir: Directory to save training outputs
            use_wandb: Whether to use Weights & Biases for logging
            wandb_project: W&B project name
            save_files: Whether to save intermediate files
        """
        self.model = model
        self.tokenizer = tokenizer
        self.save_files = save_files
        self.output_dir = Path(output_dir)
        if self.save_files:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project=wandb_project)
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        training_args: Optional[TrainingArguments] = None,
        data_collator: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Train the generation model.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            training_args: Training arguments
            data_collator: Data collator for batching
            
        Returns:
            Training results
        """
        # Set default training arguments
        if training_args is None:
            training_args = TrainingArguments(
                output_dir=str(self.output_dir),
                num_train_epochs=3,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                gradient_accumulation_steps=2,
                warmup_steps=100,
                weight_decay=0.01,
                learning_rate=5e-5,
                fp16=False,  # Disable for MPS compatibility
                logging_dir=str(self.output_dir / "logs"),
                logging_steps=10,
                eval_strategy="steps" if eval_dataset else "no",
                eval_steps=200 if eval_dataset else None,
                save_strategy="steps",
                save_steps=500,
                save_total_limit=3,
                report_to="wandb" if self.use_wandb else [],
                dataloader_pin_memory=False,
            )
        
        # Set default data collator
        if data_collator is None:
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,  # Causal LM, not masked LM
            )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model.model,  # Use the underlying model
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # Start training
        logger.info("Starting generation training...")
        train_result = trainer.train()
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Evaluate if eval dataset is provided
        eval_results = {}
        if eval_dataset:
            logger.info("Evaluating model...")
            eval_results = trainer.evaluate()
        
        results = {
            "train_results": train_result,
            "eval_results": eval_results,
        }
        
        with open(self.output_dir / "training_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Training completed. Results saved to {self.output_dir}")
        return results


class HyperparameterTuner:
    """
    Hyperparameter tuning utilities for the Enhanced LLM Model.
    """
    
    def __init__(
        self,
        model_class,
        tokenizer,
        train_dataset: Dataset,
        eval_dataset: Dataset,
    ):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            model_class: Model class to tune
            tokenizer: Tokenizer instance
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
        """
        self.model_class = model_class
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
    
    def grid_search(
        self,
        param_grid: Dict[str, List[Any]],
        metric: str = "f1",
        cv_folds: int = 3,
    ) -> Dict[str, Any]:
        """
        Perform grid search for hyperparameter tuning.
        
        Args:
            param_grid: Dictionary of parameters to search
            metric: Metric to optimize
            cv_folds: Number of cross-validation folds
            
        Returns:
            Best parameters and results
        """
        from itertools import product
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        best_score = -np.inf
        best_params = None
        all_results = []
        
        for param_combination in product(*param_values):
            params = dict(zip(param_names, param_combination))
            logger.info(f"Testing parameters: {params}")
            
            # Initialize model with current parameters
            model = self.model_class(**params)
            
            # Train and evaluate
            trainer = ClassificationTrainer(
                model=model,
                tokenizer=self.tokenizer,
                output_dir=f"./tuning/{hash(str(params))}",
            )
            
            results = trainer.train(
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
            )
            
            score = results["eval_results"].get(f"eval_{metric}", 0)
            all_results.append({
                "params": params,
                "score": score,
                "results": results,
            })
            
            if score > best_score:
                best_score = score
                best_params = params
        
        return {
            "best_params": best_params,
            "best_score": best_score,
            "all_results": all_results,
        }


class TrainingManager:
    """
    High-level training manager that orchestrates the entire training process.
    """
    
    def __init__(
        self,
        base_model_name: str = "google/gemma-3-270m-it",
        output_dir: str = "./enhanced_llm_output",
    ):
        """
        Initialize the training manager.
        
        Args:
            base_model_name: Name of the base model
            output_dir: Directory for saving outputs
        """
        self.base_model_name = base_model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def train_classification_model(
        self,
        dataset: Union[Dataset, DatasetDict],
        num_classes: int = 12,
        model_config: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Train a classification model end-to-end.
        
        Args:
            dataset: Training dataset or dataset dict
            num_classes: Number of classification classes
            model_config: Model configuration parameters
            training_config: Training configuration parameters
            
        Returns:
            Training results and model
        """
        # Initialize model
        model_config = model_config or {}
        model = TrainableLLM(
            base_model_name=self.base_model_name,
            **model_config
        )
        
        # Get tokenizer
        tokenizer = model.get_tokenizer()
        
        # Prepare datasets
        if isinstance(dataset, DatasetDict):
            train_dataset = dataset["train"]
            eval_dataset = dataset.get("validation")
            test_dataset = dataset.get("test")
        else:
            # Split dataset if it's not already split
            processor = DomainDatasetProcessor(tokenizer)
            dataset_dict = processor.split_dataset(dataset)
            train_dataset = dataset_dict["train"]
            eval_dataset = dataset_dict["validation"]
            test_dataset = dataset_dict["test"]
        
        # Initialize trainer
        trainer = ClassificationTrainer(
            model=model,
            tokenizer=tokenizer,
            output_dir=str(self.output_dir / "classification"),
        )
        
        # Train model
        results = trainer.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_args=TrainingArguments(**(training_config or {})),
        )
        
        # Test evaluation
        if test_dataset:
            test_results = trainer.evaluate_model(test_dataset)
            results["test_results"] = test_results
        
        # Save final model
        model.save_model(str(self.output_dir / "classification" / "final_model.pt"))
        
        return {
            "model": model,
            "trainer": trainer,
            "results": results,
        }
    
    def train_generation_model(
        self,
        dataset: Union[Dataset, DatasetDict],
        model_config: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Train a text generation model end-to-end.
        
        Args:
            dataset: Training dataset or dataset dict
            model_config: Model configuration parameters
            training_config: Training configuration parameters
            
        Returns:
            Training results and model
        """
        # Initialize model
        model_config = model_config or {}
        model = TextGenerationModel(
            base_model_name=self.base_model_name,
            **model_config
        )
        
        # Prepare datasets
        if isinstance(dataset, DatasetDict):
            train_dataset = dataset["train"]
            eval_dataset = dataset.get("validation")
        else:
            # Split dataset if it's not already split
            processor = DomainDatasetProcessor(model.tokenizer)
            dataset_dict = processor.split_dataset(dataset)
            train_dataset = dataset_dict["train"]
            eval_dataset = dataset_dict["validation"]
        
        # Initialize trainer
        trainer = GenerationTrainer(
            model=model,
            tokenizer=model.tokenizer,
            output_dir=str(self.output_dir / "generation"),
        )
        
        # Train model
        results = trainer.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_args=TrainingArguments(**(training_config or {})),
        )
        
        return {
            "model": model,
            "trainer": trainer,
            "results": results,
        }
