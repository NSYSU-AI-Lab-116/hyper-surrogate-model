"""
Dataset Processing and Prompt Engineering Module

This module handles dataset preprocessing and prompt construction for better
model comprehension of domain-specific data.
"""

import json
import torch
from typing import List, Dict, Any, Optional, Union, Tuple
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer
import pandas as pd
from pathlib import Path

from .utils import Logger

# Set up logger using utils.Logger
logger = Logger("dataset")


class PromptTemplate:
    """
    Template class for constructing domain-specific prompts.
    
    This class provides methods to format data into prompts that help
    the LLM better understand and process domain-specific information.
    """
    
    def __init__(self, template_type: str = "structured"):
        """Initialize prompt template."""
        self._dataset_introductions, self._structure_template = self._load_templates()
    
    def _load_templates(self) -> tuple[dict[str, str], str]:
        """
        Load predefined prompt components.
        Returns a tuple containing:
        1. A dictionary of dataset-specific introductions.
        2. A final structure template for combining components.
        """
        introductions = {
            "cifar10": """
            **Task Description**
            - **Domain**: Image Classification
            - **Benchmark**: NAS-Bench-201
            - **Dataset**: CIFAR-10
            - **Dataset Characteristics**: A standard computer vision benchmark with 10 object classes in 32x32 pixel color images. Performance on this dataset indicates a model's general capability on basic classification.
            - **Target Metric**: Final test accuracy after 200 epochs of training.
            """,
            "cifar100": """
            **Task Description**
            - **Domain**: Image Classification
            - **Benchmark**: NAS-Bench-201
            - **Dataset**: CIFAR-100
            - **Dataset Characteristics**: A more challenging benchmark with 100 fine-grained classes in 32x32 pixel color images. This tests the architecture's ability to distinguish between visually similar objects.
            - **Target Metric**: Final test accuracy after 200 epochs of training.
            """,
            "imagenet16-120": """
            **Task Description**
            - **Domain**: Image Classification
            - **Benchmark**: NAS-Bench-201
            - **Dataset**: ImageNet16-120
            - **Dataset Characteristics**: A computationally efficient but difficult benchmark with 120 diverse classes in heavily down-sized 16x16 pixel images. This tests an architecture's performance under information-constrained conditions.
            - **Target Metric**: Final test accuracy after 200 epochs of training.
            """}
        structure = """{introduction}
        **Architecture String**
        {architecture_string}
        """
        return introductions, structure
    
    def format_prompt(
        self,
        dataset_key: str,
        architecture_string: str
    ) -> str:
        """
        Format a prompt using a dataset-specific introduction.
        
        Args:
            dataset_key: The key for the dataset (e.g., 'cifar10').
            architecture_string: The string describing the architecture.
            
        Returns:
            Formatted prompt string.
        """
        if dataset_key not in self._dataset_introductions:
            raise ValueError(f"Unknown dataset_key: '{dataset_key}'. "
                            f"Please add it to PromptTemplate in dataset.py")
        
        introduction = self._dataset_introductions[dataset_key]
        
        return self._structure_template.format(
            introduction=introduction.strip(),
            architecture_string=architecture_string
        )
    
    def add_custom_template(self, name: str, template: str) -> None:
        """
        Add a custom prompt template.
        
        Args:
            name: Name of the template
            template: Template string with placeholders
        """
        self.templates[name] = template
        logger.info(f"Added custom template: {name}")


class DomainDatasetProcessor:
    """
    Processor for domain-specific datasets with prompt engineering capabilities.
    
    This class handles loading, preprocessing, and prompt construction for
    various types of domain-specific data.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        prompt_template: Optional[PromptTemplate] = None,
    ):
        """
        Initialize the dataset processor.
        
        Args:
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
            prompt_template: Custom prompt template instance
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template or PromptTemplate()
    
    def load_dataset_from_file(
        self,
        file_path: Union[str, Path],
        file_format: str = "auto"
    ) -> Dataset:
        """
        Load dataset from various file formats.
        
        Args:
            file_path: Path to the dataset file
            file_format: Format of the file ("json", "csv", "jsonl", "auto")
            
        Returns:
            Loaded dataset
        """
        file_path = Path(file_path)
        
        if file_format == "auto":
            file_format = file_path.suffix.lower().lstrip(".")
        
        logger.info(f"Loading dataset from {file_path} (format: {file_format})")
        
        if file_format == "json":
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif file_format == "jsonl":
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
        elif file_format == "csv":
            df = pd.read_csv(file_path)
            data = df.to_dict('records')
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        return Dataset.from_list(data)
    
    def create_generation_dataset(
        self,
        texts: List[str],
        labels: List[Union[str, int]],
        max_length: int = 512,
        template_type: str = "generation",
    ) -> Dataset:
        """
        Create a generation dataset with prompts.
        
        Args:
            texts: List of input texts
            labels: List of corresponding labels
            max_length: Maximum sequence length
            template_type: Type of prompt template to use
            
        Returns:
            Processed dataset ready for training
        """
        processed_data = []
        
        for text, label in zip(texts, labels):
            # Use generation prompt template
            prompt_template = PromptTemplate(template_type="generation")
            formatted_text = prompt_template.format_prompt(text)
            
            processed_data.append({
                "text": formatted_text,
                "label": label,
            })
        
        return Dataset.from_list(processed_data)
    
    def tokenize_dataset(
        self,
        dataset: Dataset,
        text_column: str = "text",
        label_column: Optional[str] = "label",
        padding: str = "max_length",
        truncation: bool = True,
    ) -> Dataset:
        """
        Tokenize the dataset for model input.
        
        Args:
            dataset: Input dataset
            text_column: Column name containing text
            label_column: Column name containing labels (optional)
            padding: Padding strategy
            truncation: Whether to truncate long sequences
            
        Returns:
            Tokenized dataset
        """
        def tokenize_function(examples):
            # 確保 tokenization 包含 attention_mask
            tokenized = self.tokenizer(
                examples[text_column],
                padding=padding,
                truncation=truncation,
                max_length=self.max_length,
                return_attention_mask=True,  # 明確返回 attention mask
                return_tensors=None  # 保持為 Python 列表格式
            )
            
            if label_column and label_column in examples:
                tokenized["labels"] = examples[label_column]
            
            return tokenized
        
        # Apply tokenization
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=[col for col in dataset.column_names 
                          if col not in ["labels"]],
        )
        
        return tokenized_dataset
    
    def split_dataset(
        self,
        dataset: Dataset,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> DatasetDict:
        """
        Split dataset into train/validation/test sets.
        
        Args:
            dataset: Input dataset
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            seed: Random seed for reproducibility
            
        Returns:
            DatasetDict with train/val/test splits
        """
        if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
            raise ValueError("Ratios must sum to 1.0")
        
        # First split: train vs (val + test)
        train_dataset = dataset.train_test_split(
            test_size=(val_ratio + test_ratio),
            seed=seed
        )
        
        # Second split: val vs test
        val_test_ratio = test_ratio / (val_ratio + test_ratio)
        val_test_dataset = train_dataset["test"].train_test_split(
            test_size=val_test_ratio,
            seed=seed
        )
        
        return DatasetDict({
            "train": train_dataset["train"],
            "validation": val_test_dataset["train"],
            "test": val_test_dataset["test"],
        })
    
    def export_dataset(
        self,
        dataset: Dataset,
        output_path: Union[str, Path],
        format: str = "json"
    ) -> None:
        """
        Export processed dataset to file.
        
        Args:
            dataset: Dataset to export
            output_path: Output file path
            format: Export format ("json", "csv", "jsonl")
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(dataset.to_list(), f, ensure_ascii=False, indent=2)
        elif format == "jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in dataset:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        elif format == "csv":
            df = pd.DataFrame(dataset.to_list())
            df.to_csv(output_path, index=False, encoding='utf-8')
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Dataset exported to {output_path}")

