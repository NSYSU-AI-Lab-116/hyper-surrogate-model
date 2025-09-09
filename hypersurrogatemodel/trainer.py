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
import torch

from .model import TrainableLLM
from .dataset import DomainDatasetProcessor
from .utils import Logger

# Set up logger using utils.Logger
logger = Logger("trainer")


class CustomDataCollatorForCausalLM:
    """Custom data collator for causal language modeling that handles variable-length sequences."""
    
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __call__(self, features):
        # Extract input_ids and labels
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        
        # Pad sequences to the same length
        max_len = min(max(len(seq) for seq in input_ids), self.max_length)
        
        # Pad input_ids
        padded_input_ids = []
        padded_labels = []
        attention_masks = []
        
        for i, (inp, lab) in enumerate(zip(input_ids, labels)):
            # Truncate if too long
            if len(inp) > max_len:
                inp = inp[:max_len]
                lab = lab[:max_len]
            
            # Create attention mask
            attention_mask = [1] * len(inp)
            
            # Pad sequences
            pad_length = max_len - len(inp)
            if pad_length > 0:
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                inp.extend([pad_token_id] * pad_length)
                lab.extend([-100] * pad_length)  # -100 is ignored in loss computation
                attention_mask.extend([0] * pad_length)
            
            padded_input_ids.append(inp)
            padded_labels.append(lab)
            attention_masks.append(attention_mask)
        
        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        }


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


class GenerationTrainer:
    """
    Trainer class for generation tasks using the Enhanced LLM Model.
    """
    
    def __init__(
        self,
        model: TrainableLLM,
        tokenizer,
        output_dir: str = "./results",
        use_wandb: bool = False,
        wandb_project: str = "hypersurrogatemodel-generation",
        save_files: bool = True,
    ):
        """
        Initialize the generation trainer.
        
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
        Train the generation model.
        
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
                remove_unused_columns=False,  # 保留所有欄位
            )
        
        # Set default data collator for generation tasks
        if data_collator is None:
            data_collator = CustomDataCollatorForCausalLM(
                tokenizer=self.tokenizer,
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
            compute_metrics=TrainingMetrics.compute_generation_metrics,
            callbacks=callbacks,
        )
        
        # Start training
        logger.info("Starting generation training...")
        train_result = trainer.train()
        
        # Save the model with improved error handling
        try:
            # First try the standard save method
            trainer.save_model()
            logger.info("Model saved successfully")
        except RuntimeError as e:
            if "share memory" in str(e) or "shared tensors" in str(e):
                logger.warning("Model save failed due to shared tensors in Gemma model")
                logger.info("This is a known issue with Gemma models where embedding and lm_head share weights")
                logger.info("The training was successful and LoRA adapters should be automatically saved")
                
                # Try creating the output directory and a simple marker file
                try:
                    from pathlib import Path
                    output_path = Path(self.output_dir)
                    output_path.mkdir(parents=True, exist_ok=True)
                    
                    # Create a marker file indicating training completion
                    marker_file = output_path / "training_completed.txt"
                    with marker_file.open('w') as f:
                        f.write("Training completed successfully\n")
                        f.write("Note: Full model save failed due to shared tensors in Gemma model\n")
                        f.write("LoRA adapters should be available in the trainer state\n")
                    logger.info(f"Training completion marker saved to {marker_file}")
                    
                except Exception as marker_error:
                    logger.warning(f"Could not create marker file: {marker_error}")
                    logger.info("Training completed successfully - model available in memory")
            else:
                raise e
        
        # Save tokenizer
        try:
            self.tokenizer.save_pretrained(self.output_dir)
            logger.info(f"Tokenizer saved to {self.output_dir}")
        except Exception as tokenizer_error:
            logger.warning(f"Failed to save tokenizer: {tokenizer_error}")
        
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
                device = next(self.model.parameters()).device
                batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                
                # predict
                logits = outputs["logits"]
                batch_predictions = torch.argmax(logits, dim=-1).cpu().numpy()
                batch_labels = batch["labels"].cpu().numpy()
                
                predictions.extend(batch_predictions)
                true_labels.extend(batch_labels)
        
        #metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        
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
