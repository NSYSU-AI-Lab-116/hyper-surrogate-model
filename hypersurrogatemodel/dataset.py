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
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptTemplate:
    """
    Template class for constructing domain-specific prompts.
    
    This class provides methods to format data into prompts that help
    the LLM better understand and process domain-specific information.
    """
    
    def __init__(self, template_type: str = "classification"):
        """
        Initialize prompt template.
        
        Args:
            template_type: Type of template (currently only "classification" is supported)
        """
        self.template_type = template_type
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load predefined prompt templates."""
        templates = {
            "classification": 
                """
                    Please analys the below data and choose the most suitable surrogate model combination for training its archtechture within its domain.
                    {text}

                    request:

                    請仔細分析上述資料的主要特徵，包含數據特徵、單位、以及結構，然後為其好澤一個最合適的替代模型架構。

                    分析過程：
                    1. 識別關鍵詞和語義特徵
                    2. 判斷情感傾向和語調
                    3. 考慮上下文語境
                    4. 做出分類決定

                    分類結果：
                """
        }
        return templates
    
    def format_prompt(
        self,
        template_type: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Format a prompt using the specified template.
        
        Args:
            template_type: Type of template to use (overrides instance default)
            **kwargs: Variables to fill in the template
            
        Returns:
            Formatted prompt string
        """
        template_type = template_type or self.template_type
        
        if template_type not in self.templates:
            raise ValueError(f"Unknown template type: {template_type}")
        
        template = self.templates[template_type]
        
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required variable for template: {e}")
    
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
    
    def create_classification_dataset(
        self,
        texts: List[str],
        labels: List[int],
        domain: str = "general",
        include_prompt: bool = True,
    ) -> Dataset:
        """
        Create a classification dataset with prompts.
        
        Args:
            texts: List of text samples
            labels: List of corresponding labels
            domain: Domain name for contextualization
            include_prompt: Whether to wrap texts in prompts
            
        Returns:
            Processed dataset
        """
        if len(texts) != len(labels):
            raise ValueError("Texts and labels must have the same length")
        
        data = []
        for text, label in zip(texts, labels):
            if include_prompt:
                # Use classification prompt template
                formatted_text = self.prompt_template.format_prompt(
                    template_type="classification",
                    text=text
                )
            else:
                formatted_text = text
            
            data.append({
                "text": formatted_text,
                "label": label,
                "domain": domain,
                "original_text": text,
            })
        
        return Dataset.from_list(data)
    
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
            # Tokenize texts
            tokenized = self.tokenizer(
                examples[text_column],
                padding="max_length",  # Use max_length padding to avoid warning
                truncation=truncation,
                max_length=self.max_length,
                return_tensors=None,
            )
            
            # Add labels if present
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
    
    def create_sample_domain_dataset(self, domain: str = "sentiment_analysis") -> Dataset:
        """
        Create a sample domain-specific dataset for demonstration.
        
        Args:
            domain: Domain type for sample data
            
        Returns:
            Sample dataset
        """
        if domain == "sentiment_analysis":
            texts = [
                "這個產品真的很棒，品質超出期待！",
                "服務態度很差，完全不推薦。",
                "價格合理，功能實用，值得購買。",
                "包裝破損，產品有瑕疵，很失望。",
                "客服回應迅速，解決問題很有效率。",
                "等了很久才收到，而且品質一般。",
                "外觀設計很漂亮，使用起來很順手。",
                "說明書不清楚，操作很複雜。",
                "物超所值，會推薦給朋友。",
                "退貨流程很麻煩，客服態度冷漠。",
            ]
            labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1: positive, 0: negative
            
            return self.create_classification_dataset(
                texts=texts,
                labels=labels,
                domain=domain,
            )
        
        else:
            raise ValueError(f"Unknown sample domain: {domain}")
    
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


class DatasetAugmentor:
    """
    Dataset augmentation utilities for improving model robustness.
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizer):
        """
        Initialize the augmentor.
        
        Args:
            tokenizer: Tokenizer for text processing
        """
        self.tokenizer = tokenizer
    
    def augment_by_paraphrasing(
        self,
        texts: List[str],
        paraphrase_templates: List[str],
    ) -> List[str]:
        """
        Augment texts by applying paraphrase templates.
        
        Args:
            texts: Original texts
            paraphrase_templates: Templates for paraphrasing
            
        Returns:
            Augmented texts
        """
        augmented = []
        for text in texts:
            for template in paraphrase_templates:
                augmented.append(template.format(text=text))
        return augmented
    
    def augment_by_noise(
        self,
        texts: List[str],
        noise_ratio: float = 0.1,
        noise_types: List[str] = ["mask", "substitute"],
    ) -> List[str]:
        """
        Add noise to texts for robustness training.
        
        Args:
            texts: Original texts
            noise_ratio: Ratio of tokens to modify
            noise_types: Types of noise to apply
            
        Returns:
            Noisy texts
        """
        # This is a simplified implementation
        # In practice, you might want to use more sophisticated methods
        augmented = []
        for text in texts:
            tokens = self.tokenizer.tokenize(text)
            num_noise = int(len(tokens) * noise_ratio)
            
            if "mask" in noise_types and num_noise > 0:
                # Replace some tokens with mask token
                noisy_tokens = tokens.copy()
                import random
                positions = random.sample(range(len(tokens)), min(num_noise, len(tokens)))
                for pos in positions:
                    mask_token = self.tokenizer.mask_token or "[MASK]"
                    # Ensure mask_token is a string
                    if isinstance(mask_token, list):
                        mask_token = mask_token[0] if mask_token else "[MASK]"
                    noisy_tokens[pos] = str(mask_token)
                augmented.append(self.tokenizer.convert_tokens_to_string(noisy_tokens))
        
        return augmented
