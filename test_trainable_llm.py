"""
Test script for the cleaned TrainableLLM model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize logger
from hypersurrogatemodel.utils import Logger
logger = Logger("test_trainable_llm")

def test_trainable_llm():
    """Test the TrainableLLM model"""
    try:
        from hypersurrogatemodel import TrainableLLM
        
        logger.success("✅ Successfully imported TrainableLLM")
        
        # Initialize model
        model = TrainableLLM(
            base_model_name="google/gemma-3-270m-it",
            use_lora=True
        )
        
        logger.success("✅ Successfully initialized TrainableLLM")
        
        # Test tokenizer
        tokenizer = model.get_tokenizer()
        logger.success(f"✅ Tokenizer loaded: {type(tokenizer).__name__}")
        
        # Test model info
        info = model.get_model_info()
        logger.success(f"✅ Model info:")
        logger.info(f"   - Base model: {info['base_model']}")
        logger.info(f"   - Total parameters: {info['total_parameters']:,}")
        logger.info(f"   - Trainable parameters: {info['trainable_parameters']:,}")
        logger.info(f"   - Trainable ratio: {info['trainable_ratio']:.2f}%")
        
        # Test text generation
        prompt = "Hello, this is a test"
        generated = model.generate_text(prompt, max_new_tokens=20)
        logger.success(f"✅ Text generation test:")
        logger.info(f"   - Prompt: {prompt}")
        logger.info(f"   - Generated: {generated}")
        
        logger.success("\n🎉 All tests passed! TrainableLLM is working correctly.")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_trainable_llm()
