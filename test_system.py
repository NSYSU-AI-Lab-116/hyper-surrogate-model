"""
AI generated test script for the Enhanced LLM Model system.
"""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class Color:
    """Simple color class for terminal output."""
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    

def test_imports():
    """Test that all modules can be imported successfully."""
    print("Testing imports...")
    
    try:
        from hypersurrogatemodel import (
            EnhancedLLMModel,
            TextGenerationModel,
            DomainDatasetProcessor,
            PromptTemplate,
            ClassificationTrainer,
            ModelEvaluator,
            set_random_seed,
            get_device,
            Logger
        )
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_initialization():
    """Test model initialization."""
    print("\nTesting model initialization...")
    
    try:
        from hypersurrogatemodel import EnhancedLLMModel, set_random_seed
        
        set_random_seed(42)
        
        # Test with smaller model for quick testing
        model = EnhancedLLMModel(
            base_model_name="google/gemma-3-270m-it",
            num_classes=12,
            hidden_size=128,  # Smaller for testing
            dropout_rate=0.1,
            use_lora=True,
        )
        
        model_info = model.get_model_info()
        print(f"✅ Model initialized successfully")
        print(f"   - Total parameters: {model_info['total_parameters']:,}")
        print(f"   - Trainable parameters: {model_info['trainable_parameters']:,}")
        print(f"   - Trainable ratio: {model_info['trainable_ratio']:.2f}%")
        
        return True, model
    except Exception as e:
        print(f"❌ Model initialization error: {e}")
        return False, None

def test_dataset_processing():
    """Test dataset processing functionality."""
    print("\nTesting dataset processing...")
    
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
        
        print(f"✅ Dataset processing successful")
        print(f"   - Created dataset with {len(dataset)} samples")
        print(f"   - Prompt template working")
        
        return True
    except Exception as e:
        print(f"❌ Dataset processing error: {e}")
        return False

def test_utilities():
    """Test utility functions."""
    print("\nTesting utilities...")
    
    try:
        from hypersurrogatemodel import get_device, get_system_info, Logger
        
        # Test device detection
        device = get_device()
        print(f"✅ Device detection: {device}")
        
        # Test system info
        sys_info = get_system_info()
        print(f"✅ System info collected")
        print(f"   - CPU count: {sys_info['cpu_count']}")
        print(f"   - Memory total: {sys_info['memory_total_gb']:.1f} GB")
        print(f"   - CUDA available: {sys_info['cuda_available']}")
        print(f"   - MPS available: {sys_info['mps_available']}")
        
        # Test logger
        logger = Logger("test_logger", log_dir="./test_logs")
        logger.info("Test log message")
        print("✅ Logger working")
        
        return True
    except Exception as e:
        print(f"❌ Utilities error: {e}")
        return False

def test_model_forward_pass():
    """Test model forward pass with dummy data."""
    print("\nTesting model forward pass...")
    
    try:
        from hypersurrogatemodel import EnhancedLLMModel
        from hypersurrogatemodel.utils import get_device
        
        model = EnhancedLLMModel(
            base_model_name="google/gemma-3-270m-it",
            num_classes=12,
            hidden_size=128,
            use_lora=True,
        )
        
        tokenizer = model.get_tokenizer()
        device = get_device()
        model = model.to(device)
        
        # Test input
        test_text = "This is a test input for the model."
        inputs = tokenizer(
            test_text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        input_ids = inputs["input_ids"].to(device) # type: ignore
        attention_mask = inputs["attention_mask"].to(device) # type: ignore
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        logits = outputs["logits"]
        print(f"✅ Forward pass successful")
        print(f"   - Input IDs shape: {input_ids.shape}")
        print(f"   - Output logits shape: {logits.shape}")
        print(f"   - Expected output shape: [1, 12]")
        
        # Check output shape
        assert logits.shape == (1, 12), f"Expected shape [1, 12], got {logits.shape}"
        print("✅ Output shape verification passed")
        
        return True
    except Exception as e:
        print(f"❌ Forward pass error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Enhanced LLM Model - Quick Test Suite")
    print("=" * 60)
    
    test_results = []
    
    # Run tests
    test_results.append(("Imports", test_imports()))
    test_results.append(("Model Initialization", test_model_initialization()[0]))
    test_results.append(("Dataset Processing", test_dataset_processing()))
    test_results.append(("Utilities", test_utilities()))
    test_results.append(("Model Forward Pass", test_model_forward_pass()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = f"{Color.GREEN}PASS{Color.RESET}" if result else f"{Color.RED}FAIL{Color.RESET}"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print(f"{Color.GREEN}All tests passed! The system is ready to use.{Color.RESET}")
        return True
    else:
        print(f"{Color.RED}Some tests failed. Please check the errors above.{Color.RESET}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
