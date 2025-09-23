"""
Evaluation Interface Module

This module provides basic evaluation capabilities for the Enhanced LLM Model.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report
)
from hypersurrogatemodel.config import config
import json
from datetime import datetime

from .model import TrainableLLM
from .utils import Logger

# Set up logger using utils.Logger
logger = Logger("evaluator")


class ModelEvaluator:
    """
    Basic model evaluation functionality.
    """
    
    def __init__(
        self,
        model: TrainableLLM,
        tokenizer,
        class_names: Optional[List[str]] = None,
    ):
        """
        Initialize the evaluator.
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer for text processing
            class_names: Names of classes for classification tasks
        """
        self.model = model
        self.tokenizer = tokenizer
        self.class_names = class_names or [f"Class_{i}" for i in range(12)]
        
        # Set model to evaluation mode
        self.model.eval()
    
    def evaluate_classification(
        self,
        batch_size: int = 8,
        save_results: bool = True,
        output_dir: Union[str, Path] = "./evaluation_results",
    ) -> Dict[str, Any]:
        """
        Basic evaluation for classification tasks.
        
        Args:
            test_data: List of test samples with "text" and "label" keys
            batch_size: Batch size for evaluation
            save_results: Whether to save detailed results
            output_dir: Directory to save results
            
        Returns:
            Basic evaluation results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if (config.dataset.test_data_path is None) or (not Path(config.dataset.test_data_path).is_file()):
            logger.error("Test data path is not specified or the file does not exist.")
            raise FileNotFoundError("Test data file not found.")
        with open(config.dataset.test_data_path, 'r') as f:
            test_data = json.load(f)
        
        predictions = []
        true_labels = []
        
        logger.info("Starting classification evaluation...")
        
        # Process in batches
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i + batch_size]
            batch_texts = [item["text"] for item in batch]
            batch_labels = [item["label"] for item in batch]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
                
                logits = outputs["logits"]
                batch_predictions = torch.argmax(logits, dim=-1)
                
                predictions.extend(batch_predictions.cpu().numpy())
                true_labels.extend(batch_labels)
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        
        # Basic metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # Classification report
        report = classification_report(
            true_labels, predictions,
            target_names=self.class_names,
            output_dict=True
        )
        
        # Compile results
        results = {
            "basic_metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            },
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "total_samples": len(test_data),
            "correct_predictions": int(np.sum(predictions == true_labels)),
            "incorrect_predictions": int(np.sum(predictions != true_labels)),
        }
        
        if save_results:
            # Save results
            with open(output_path / "evaluation_results.json", 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Evaluation results saved to {output_path}")
        
        logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        return results