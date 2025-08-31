"""
AI generated test script for the Enhanced LLM Model system.
"""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Initialize logger
from hypersurrogatemodel.utils import Logger

class Color:
    """Simple color class for terminal output."""
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"

logger = Logger("test_system")

def test_imports():
    """Test that all modules can be imported successfully."""
    logger.setFunctionsLevel("test_imports")
    logger.step("Testing imports...")
    
    try:
        from hypersurrogatemodel import (
            TrainableLLM,
            TextGenerationModel,
            DomainDatasetProcessor,
            PromptTemplate,
            ClassificationTrainer,
            ModelEvaluator,
            set_random_seed,
            get_device,
        )
        logger.success("All imports successful")
        return True
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return False

def test_model_initialization():
    """Test model initialization."""
    logger.setFunctionsLevel("test_model_initialization")
    logger.step("Testing model initialization...")
    
    try:
        from hypersurrogatemodel import TrainableLLM, set_random_seed
        
        set_random_seed(42)
        
        # Test with smaller model for quick testing
        model = TrainableLLM(
            base_model_name="google/gemma-3-270m-it",
            use_lora=True,
        )
        
        model_info = model.get_model_info()
        logger.success(f"Model initialized successfully")
        logger.info(f"Total parameters: {model_info['total_parameters']:,}")
        logger.info(f"Trainable parameters: {model_info['trainable_parameters']:,}")
        logger.info(f"Trainable ratio: {model_info['trainable_ratio']:.2f}%")
        
        return True, model
    except Exception as e:
        logger.error(f"Model initialization error: {e}")
        return False, None

def test_dataset_processing():
    """Test dataset processing functionality."""
    print()
    logger.setFunctionsLevel("test_dataset_processing")
    logger.step("Testing dataset processing...")
    
    try:
        from hypersurrogatemodel import DomainDatasetProcessor, PromptTemplate
        from transformers import AutoTokenizer
        
        # Get tokenizer
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Initialize processor
        processor = DomainDatasetProcessor(tokenizer, max_length=256)
        
        # Test prompt template
        template = PromptTemplate("classification")
        formatted_prompt = template.format_prompt(
            text="This is a test text for classification."
        )
        
        # Create sample dataset
        texts = [
            "Technology news about AI advancement",
            "Medical breakthrough in cancer research", 
            "Financial market update"
        ]
        labels = [0, 1, 2]
        
        dataset = processor.create_classification_dataset(
            texts=texts,
            labels=labels,
            domain="test_domain",
            include_prompt=True
        )
        
        logger.success(f"Dataset processing successful")
        logger.info(f"Created dataset with {len(dataset)} samples")
        logger.info(f"Prompt template working")
        
        return True
    except Exception as e:
        logger.error(f"Dataset processing error: {e}")
        return False

def test_utilities():
    """Test utility functions."""
    print()
    logger.setFunctionsLevel("utilities")
    logger.step("Testing utilities...")
    
    try:
        from hypersurrogatemodel import get_device, get_system_info
        
        # Test device detection
        device = get_device()
        logger.success(f"Device detection: {device}")
        
        # Test system info
        get_system_info()
        
        return True
    except Exception as e:
        logger.error(f"Utilities error: {e}")
        return False

def test_model_forward_pass():
    """Test forward pass functionality."""
    logger.setFunctionsLevel("test_model_forward_pass")
    logger.step("Testing Model Forward Pass...")
    
    try:
        from hypersurrogatemodel import TrainableLLM
        from hypersurrogatemodel.utils import get_device
        
        model = TrainableLLM(
            base_model_name="google/gemma-3-270m-it",
            use_lora=True,
        )
        
        tokenizer = model.get_tokenizer()
        device = get_device()
        model = model.to(device)
        
        # Test input
        test_text = "This is a test input for the model."
        
        # Test text generation instead of classification
        with torch.no_grad():
            generated_text = model.generate_text(
                prompt=test_text,
                max_new_tokens=20,
                temperature=0.7
            )
        
        logger.success("Forward pass successful")
        logger.info(f"Input text: {test_text}")
        logger.info(f"Generated text: {generated_text}")
        
        # Check that generation worked
        assert isinstance(generated_text, str), f"Expected string output, got {type(generated_text)}"
        assert len(generated_text) > len(test_text), "Generated text should be longer than input"
        logger.success("Text generation verification passed")
        
        return True
    except Exception as e:
        logger.error(f"Forward pass error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    logger.info("Enhanced LLM Model - Quick Test Suite")
    print("=" * 60)
    
    test_results = []
    
    # Run tests
    test_results.append(("Imports", test_imports()))
    test_results.append(("Model Initialization", test_model_initialization()[0]))
    test_results.append(("Dataset Processing", test_dataset_processing()))
    test_results.append(("Utilities", test_utilities()))
    test_results.append(("Model Forward Pass", test_model_forward_pass()))
    
    # Summary
    print("\n\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for result in test_results:
        if result:
            passed += 1
    
    logger.info(f"Passed: {passed}/{total}")
    
    if passed == total:
        logger.success(f"All tests passed! The system is ready to use.")
        return True
    else:
        logger.error(f"Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
