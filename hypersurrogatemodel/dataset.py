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
        """
        Initialize prompt template.
        
        Args:
            template_type: "generation")
        """
        self.template_type = template_type
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load predefined prompt templates."""
        templates = {
            "generation": 
                """
                    Please analys the below data and choose the most suitable surrogate model combination for training its archtechture within its domain.
                    {text}
                """,
            "structured":
                """
                    Please analys the below data and give the most suitable accuracy of the structure, indicating its performance after trained.
                    {text}
                    
                    Next, output the result with a structured JSON format as follows:
                    {"accuracy":double}
                    
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

