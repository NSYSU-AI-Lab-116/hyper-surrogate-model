import sys
import os
from hypersurrogatemodel import Logger
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    from hypersurrogatemodel import TrainableLLM
    logger = Logger("llm-pipeline", use_colors=True)
    logger.info("TrainableLLM Example")
    logger.info("=" * 50)
    
    # Initialize the model
    logger.step("Initializing TrainableLLM...")
    model = TrainableLLM(
        base_model_name="google/gemma-3-270m-it",
        use_lora=True  
    )
    
    # Show model info
    info = model.get_model_info()
    logger.success(f"Model loaded successfully!")
    logger.info(f"   - Base model: {info['base_model']}")
    logger.info(f"   - Total parameters: {info['total_parameters']:,}")
    logger.info(f"   - Trainable parameters: {info['trainable_parameters']:,}")
    logger.info(f"   - Trainable ratio: {info['trainable_ratio']:.2f}%")
    logger.info("")
    
    # Text generation examples
    prompts = [
        "Artificial intelligence is",
        "The future of technology will",
        "Machine learning helps us",
    ]
    
    logger.step("Text Generation Examples:")
    logger.info("-" * 30)
    
    for i, prompt in enumerate(prompts, 1):
        logger.info(f"{i}. Prompt: {prompt}")
        generated = model.generate_text(
            prompt, 
            max_new_tokens=30,
            temperature=0.7,
            do_sample=True
        )
        logger.info(f"   Response: {generated}")
        logger.info("")
    
    logger.success("✨ The model is ready for training/fine-tuning!")
    logger.info("   - Use model.forward() for training with loss calculation")
    logger.info("   - Use model.generate_text() for inference")
    logger.info("   - Use model.save_model() to save fine-tuned weights")

if __name__ == "__main__":
    main()
