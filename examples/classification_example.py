"""
Classification Training Example

This example demonstrates how to train the Enhanced LLM Model for classification tasks
with a 12-dimensional output as specified in the requirements.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hypersurrogatemodel import (
    EnhancedLLMModel,
    DomainDatasetProcessor,
    ClassificationTrainer,
    TrainingManager,
    ModelEvaluator,
    set_random_seed,
    create_experiment_directory,
    Logger
)
from transformers import TrainingArguments


def main():
    """
    Main function demonstrating classification training workflow.
    """
    # Initialize logger
    logger = Logger("classification_example")
    logger.info("Starting classification training example")
    
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Create experiment directory
    experiment_dir = create_experiment_directory(
        base_dir="./experiments",
        experiment_name="classification_12_class",
        timestamp=True
    )
    
    # Step 1: Initialize the Enhanced LLM Model (Model A) with 12 output dimensions
    logger.info("Initializing Enhanced LLM Model with 12 classes...")
    model = EnhancedLLMModel(
        base_model_name="google/gemma-3-270m-it",
        num_classes=12,  # Required 12 dimensions output
        hidden_size=256,
        dropout_rate=0.1,
        use_lora=True,
    )
    
    # Get tokenizer
    tokenizer = model.get_tokenizer()
    
    # Print model information
    model_info = model.get_model_info()
    logger.info(f"Model initialized with {model_info['total_parameters']:,} total parameters")
    logger.info(f"Trainable parameters: {model_info['trainable_parameters']:,} ({model_info['trainable_ratio']:.2f}%)")
    
    # Step 2: Prepare domain-specific dataset with prompt engineering
    logger.info("Preparing domain-specific dataset...")
    dataset_processor = DomainDatasetProcessor(tokenizer=tokenizer, max_length=512)
    
    # Create sample 12-class classification dataset
    # You can replace this with your actual domain data
    sample_data = create_sample_12_class_dataset()
    
    # Create dataset with prompt engineering
    dataset = dataset_processor.create_classification_dataset(
        texts=sample_data["texts"],
        labels=sample_data["labels"],
        domain="multi_class_classification",
        include_prompt=True,  # Enable prompt engineering
    )
    
    # Tokenize the dataset
    tokenized_dataset = dataset_processor.tokenize_dataset(
        dataset=dataset,
        text_column="text",
        label_column="label",
    )
    
    # Split into train/val/test
    dataset_dict = dataset_processor.split_dataset(
        dataset=tokenized_dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    
    logger.info(f"Dataset split - Train: {len(dataset_dict['train'])}, "
                f"Val: {len(dataset_dict['validation'])}, "
                f"Test: {len(dataset_dict['test'])}")
    
    # Step 3: Set up training configuration
    training_args = TrainingArguments(
        output_dir=str(experiment_dir / "model_output"),
        num_train_epochs=5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        warmup_steps=100,
        weight_decay=0.01,
        learning_rate=2e-5,
        fp16=False,  # Disabled for MPS compatibility
        logging_dir=str(experiment_dir / "logs"),
        logging_steps=10,
        evaluation_strategy="steps", # type: ignore
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        report_to=[],  # Disable wandb for this example
        dataloader_pin_memory=False,
    )
    
    # Step 4: Initialize trainer and start training
    logger.info("Starting training...")
    trainer = ClassificationTrainer(
        model=model,
        tokenizer=tokenizer,
        output_dir=str(experiment_dir / "training_output"),
        use_wandb=False,
    )
    
    # Train the model
    training_results = trainer.train(
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        training_args=training_args,
    )
    
    logger.info("Training completed!")
    logger.info(f"Final training loss: {training_results['train_results'].training_loss:.4f}")
    
    if training_results['eval_results']:
        logger.info(f"Final validation F1: {training_results['eval_results']['eval_f1']:.4f}")
        logger.info(f"Final validation accuracy: {training_results['eval_results']['eval_accuracy']:.4f}")
    
    # Step 5: Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate_model(
        test_dataset=dataset_dict["test"],
        batch_size=8,
    )
    
    logger.info(f"Test accuracy: {test_results['accuracy']:.4f}")
    logger.info(f"Test F1 score: {test_results['f1']:.4f}")
    
    # Step 6: Save the final model
    model_save_path = experiment_dir / "final_model.pt"
    model.save_model(str(model_save_path))
    logger.info(f"Model saved to: {model_save_path}")
    
    # Step 7: Demonstrate inference
    logger.info("Demonstrating inference...")
    demonstrate_inference(model, tokenizer, sample_data["class_names"])
    
    logger.info("Classification training example completed successfully!")
    return {
        "model": model,
        "training_results": training_results,
        "test_results": test_results,
        "experiment_dir": experiment_dir,
    }


def create_sample_12_class_dataset():
    """
    Create a sample 12-class dataset for demonstration.
    
    Returns:
        Dictionary containing texts, labels, and class names
    """
    # Define 12 classes for demonstration
    class_names = [
        "Technology", "Healthcare", "Finance", "Education",
        "Entertainment", "Sports", "Politics", "Science",
        "Travel", "Food", "Fashion", "Environment"
    ]
    
    # Sample data for each class
    sample_texts = [
        # Technology (0)
        "The latest smartphone features advanced AI capabilities and 5G connectivity.",
        "Machine learning algorithms are revolutionizing data analysis.",
        "Cloud computing provides scalable infrastructure for businesses.",
        
        # Healthcare (1)
        "New medical research shows promising results for cancer treatment.",
        "Telemedicine is improving access to healthcare services.",
        "Mental health awareness is becoming increasingly important.",
        
        # Finance (2)
        "Cryptocurrency markets are experiencing significant volatility.",
        "Digital banking services are transforming financial transactions.",
        "Investment strategies should focus on long-term growth.",
        
        # Education (3)
        "Online learning platforms are changing education delivery methods.",
        "STEM education is crucial for future workforce development.",
        "Educational technology enhances student engagement.",
        
        # Entertainment (4)
        "Streaming services are dominating the entertainment industry.",
        "Video games are becoming more immersive with VR technology.",
        "Movie theaters are adapting to digital transformation.",
        
        # Sports (5)
        "Olympic athletes showcase incredible human performance.",
        "Professional sports leagues are implementing new safety protocols.",
        "Sports analytics is revolutionizing team strategies.",
        
        # Politics (6)
        "Government policies are addressing climate change concerns.",
        "Democratic processes require citizen participation.",
        "International relations affect global stability.",
        
        # Science (7)
        "Scientific research contributes to human knowledge advancement.",
        "Space exploration missions discover new celestial phenomena.",
        "Climate science provides insights into environmental changes.",
        
        # Travel (8)
        "International travel restrictions are gradually being lifted.",
        "Sustainable tourism practices protect natural environments.",
        "Cultural exchange through travel enriches personal experiences.",
        
        # Food (9)
        "Organic farming practices promote environmental sustainability.",
        "Culinary traditions reflect cultural heritage and identity.",
        "Nutritious diets contribute to overall health and wellbeing.",
        
        # Fashion (10)
        "Sustainable fashion brands are gaining consumer popularity.",
        "Fashion trends reflect social and cultural movements.",
        "Textile innovation creates new fabric technologies.",
        
        # Environment (11)
        "Renewable energy sources reduce carbon emissions.",
        "Conservation efforts protect endangered species.",
        "Climate action requires global cooperation and commitment.",
    ]
    
    # Create labels for the texts
    labels = []
    for i, class_name in enumerate(class_names):
        labels.extend([i] * 3)  # 3 samples per class
    
    return {
        "texts": sample_texts,
        "labels": labels,
        "class_names": class_names,
    }


def demonstrate_inference(model, tokenizer, class_names):
    """
    Demonstrate model inference on new examples.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        class_names: List of class names
    """
    # Set model to evaluation mode
    model.eval()
    
    # Test examples
    test_examples = [
        "Artificial intelligence is transforming software development.",
        "The doctor recommended a new treatment for the patient.",
        "Stock market indices reached new record highs.",
        "Students are adapting to hybrid learning environments.",
    ]
    
    print("\n" + "="*60)
    print("INFERENCE DEMONSTRATION")
    print("="*60)
    
    for text in test_examples:
        # Create prompt using the same format as training
        from hypersurrogatemodel.dataset import PromptTemplate
        prompt_template = PromptTemplate("classification")
        formatted_text = prompt_template.format_prompt(text=text)
        
        # Tokenize
        inputs = tokenizer(
            formatted_text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        
        # Get prediction
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            
            logits = outputs["logits"]
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(logits, dim=-1).item()
            confidence = probabilities[0][predicted_class].item() # type: ignore
        
        print(f"\nInput: {text}")
        print(f"Predicted Class: {class_names[predicted_class]}")
        print(f"Confidence: {confidence:.3f}")
        
        # Show top 3 predictions
        top_probs, top_indices = torch.topk(probabilities[0], 3)
        print("Top 3 predictions:")
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            print(f"  {i+1}. {class_names[idx]}: {prob:.3f}")


if __name__ == "__main__":
    import torch
    
    # Check if required packages are available
    try:
        results = main()
        print(f"\nExperiment completed successfully!")
        print(f"Results saved to: {results['experiment_dir']}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
