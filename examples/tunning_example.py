"""
Complete Example: Dataset Comparison and Adaptive Tuning

This example demonstrates how to:
1. Load a dataset with inputs and expected answers
2. Generate predictions using the LLM
3. Compare predictions with ground truth
4. Perform adaptive tuning based on differences
"""

import json
import numpy as np
from pathlib import Path
from hypersurrogatemodel import (
    TrainableLLM, 
    ComparisonTuner,
    ConfigManager,
    Logger,
    set_random_seed
)

def create_sample_dataset():
    """Create a sample dataset for demonstration."""
    sample_data = [
        {
            "text": "Design a surrogate model for aerodynamic simulation with 1000 data points, 5 input variables, and requires high accuracy.",
            "answer": "neural_network"
        },
        {
            "text": "Need surrogate model for structural optimization with 500 samples, 3 design variables, budget is limited.",
            "answer": "polynomial_regression"
        },
        {
            "text": "Complex nonlinear system with 2000 data points, 8 input variables, need uncertainty quantification.",
            "answer": "gaussian_process"
        },
        {
            "text": "Simple relationship between 2 variables, 200 data points, need fast prediction.",
            "answer": "linear_regression"
        },
        {
            "text": "High-dimensional problem with 50 variables, 10000 data points, computational efficiency is critical.",
            "answer": "random_forest"
        },
        {
            "text": "Optimization problem with discontinuous response, 800 data points, 6 variables.",
            "answer": "support_vector_machine"
        },
        {
            "text": "Time series prediction for engineering system, 1500 data points, temporal dependencies.",
            "answer": "neural_network"
        },
        {
            "text": "Multi-objective optimization, 300 data points, 4 variables, need interpretability.",
            "answer": "polynomial_regression"
        }
    ]
    
    # Save file
    dataset_path = Path("sample_surrogate_dataset.json")
    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print(f"Sample dataset created: {dataset_path}")
    return dataset_path

def main():
    """Main function demonstrating the complete workflow."""
    logger = Logger("comparison_tuning_example")
    set_random_seed(42)
    
    logger.info("Starting dataset comparison and adaptive tuning example")
    
    # just sample data
    # Todo: remove sample
    dataset_path = create_sample_dataset()
    
    # Init
    logger.info("Initializing TrainableLLM model...")
    model = TrainableLLM(
        base_model_name="google/gemma-3-270m-it",
        use_lora=True,
        lora_config={
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "target_modules": ["q_proj", "v_proj"],
        }
    )
    
    tokenizer = model.get_tokenizer()
    
    # Init comparison tuner
    logger.info("Initializing ComparisonTuner...")
    tuner = ComparisonTuner(
        model=model,
        tokenizer=tokenizer,
        output_dir="./comparison_tuning_results",
        use_wandb=False,  # Set to True if you want to use W&B
        save_files=False
    )
    
    print("\n" + "="*60)
    print("STEP 1: Initial Comparison with Dataset")
    print("="*60)
    
    # Load dataset and compare with current model
    logger.info("Loading dataset and generating initial predictions...")
    initial_results = tuner.load_and_compare_dataset(
        dataset_path=dataset_path,
        text_column="text",
        answer_column="answer",
        task_type="generation",  # Using generation for text-to-text prediction
        comparison_method="similarity"  # Use similarity for text comparison
    )
    
    # Print initial results
    print(f"\\nInitial Performance:")
    print(f"Accuracy: {initial_results['overall_metrics']['accuracy']:.3f}")
    print(f"Average Similarity: {initial_results['overall_metrics']['average_similarity']:.3f}")
    print(f"Correct Predictions: {initial_results['overall_metrics']['correct_predictions']}")
    print(f"Incorrect Predictions: {initial_results['overall_metrics']['incorrect_predictions']}")
    
    print(f"\\nError Analysis:")
    error_analysis = initial_results['error_analysis']
    print(f"Error Types: {error_analysis['error_types']}")
    if error_analysis['most_common_error']:
        print(f"Most Common Error: {error_analysis['most_common_error']}")
    
    # Show some examples of differences
    print(f"\\nExample Differences (showing first 3):")
    for i, diff in enumerate(initial_results['differences'][:3]):
        print(f"\\nExample {i+1}:")
        print(f"Input: {diff['input_text'][:100]}...")
        print(f"Predicted: {diff['prediction']}")
        print(f"Expected: {diff['ground_truth']}")
        print(f"Similarity: {diff['similarity']:.3f}")
    
    print("\\n" + "="*60)
    print("STEP 2: Adaptive Tuning Based on Differences")
    print("="*60)
    
    # Perform adaptive tuning if there are errors
    if initial_results['overall_metrics']['incorrect_predictions'] > 0:
        logger.info("Starting adaptive tuning...")
        
        # Try different tuning strategies
        strategies = ["error_focused", "incremental"]
        
        for strategy in strategies:
            print(f"\\n--- Trying {strategy} strategy ---")
            
            tuning_results = tuner.adaptive_tuning(
                comparison_results=initial_results,
                dataset_path=dataset_path,
                text_column="text",
                answer_column="answer",
                tuning_strategy=strategy,
                max_epochs=2,  # Keep it small for demonstration
                learning_rate=1e-5,
            )
            
            print(f"\\nTuning Results for {strategy}:")
            print(f"Training Data Size: {tuning_results['training_data_size']}")
            
            pre_metrics = tuning_results['pre_tuning_metrics']
            post_metrics = tuning_results['post_tuning_metrics']
            
            print(f"\\nBefore Tuning:")
            print(f"  Accuracy: {pre_metrics['accuracy']:.3f}")
            print(f"  Avg Similarity: {pre_metrics['average_similarity']:.3f}")
            
            print(f"\\nAfter Tuning:")
            print(f"  Accuracy: {post_metrics['accuracy']:.3f}")
            print(f"  Avg Similarity: {post_metrics['average_similarity']:.3f}")
            
            # Show improvement analysis
            improvements = tuning_results['improvement_analysis']['improvements']
            print(f"\\nImprovements:")
            for metric, improvement in improvements.items():
                print(f"  {metric}: {improvement['absolute_improvement']:+.3f} "
                     f"({improvement['percentage_improvement']:+.1f}%)")
            
            error_reduction = tuning_results['improvement_analysis']['error_reduction']
            print(f"  Error Reduction: {error_reduction} samples")
            
            # Break after first successful strategy for demo
            if error_reduction > 0:
                print(f"\\n✅ {strategy} strategy showed improvement!")
                break
        
        print("\\n" + "="*60)
        print("STEP 3: Final Evaluation")
        print("="*60)
        
        # Final evaluation to see overall improvement
        final_results = tuner.load_and_compare_dataset(
            dataset_path=dataset_path,
            text_column="text",
            answer_column="answer",
            task_type="generation",
            comparison_method="similarity"
        )
        
        print(f"\\nFinal Performance:")
        print(f"Accuracy: {final_results['overall_metrics']['accuracy']:.3f}")
        print(f"Average Similarity: {final_results['overall_metrics']['average_similarity']:.3f}")
        print(f"Correct Predictions: {final_results['overall_metrics']['correct_predictions']}")
        print(f"Incorrect Predictions: {final_results['overall_metrics']['incorrect_predictions']}")
        
        # Compare with initial
        acc_improvement = final_results['overall_metrics']['accuracy'] - initial_results['overall_metrics']['accuracy']
        sim_improvement = final_results['overall_metrics']['average_similarity'] - initial_results['overall_metrics']['average_similarity']
        
        print(f"\\nOverall Improvement:")
        print(f"Accuracy: {acc_improvement:+.3f}")
        print(f"Average Similarity: {sim_improvement:+.3f}")
        
    else:
        print("\\n✅ Model already performs perfectly on this dataset!")
    
    print("\\n" + "="*60)
    print("STEP 4: Save Model and Results")
    print("="*60)
    
    # Save the tuned model
    model_save_path = "./tuned_surrogate_model"
    model.save_model(model_save_path)
    print(f"Tuned model saved to: {model_save_path}")
    
    # Save configuration
    config = {
        "model_config": {
            "base_model_name": "google/gemma-3-270m-it",
            "use_lora": True,
            "lora_config": {
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.1,
                "target_modules": ["q_proj", "v_proj"],
            }
        },
        "tuning_config": {
            "comparison_method": "similarity",
            "task_type": "generation",
            "max_epochs": 2,
            "learning_rate": 1e-5,
        },
        "dataset_info": {
            "path": str(dataset_path),
            "samples": 8,
            "task": "surrogate_model_selection"
        }
    }
    
    config_path = Path("./tuning_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to: {config_path}")
    
    print("\\n" + "="*60)
    print("Example completed successfully!")
    print("="*60)
    
    print(f"\\nGenerated files:")
    print(f"- Dataset: {dataset_path}")
    print(f"- Model: {model_save_path}")
    print(f"- Config: {config_path}")
    print(f"- Results: ./comparison_tuning_results/")
    
    logger.info("Example completed successfully")

def demonstrate_classification_tuning():
    """Demonstrate classification-based tuning."""
    print("\\n" + "="*60)
    print("BONUS: Classification Task Example")
    print("="*60)
    
    # Create classification dataset
    classification_data = [
        {"text": "High-accuracy aerodynamic simulation", "answer": 0},  # neural_network
        {"text": "Simple linear relationship", "answer": 1},           # linear_regression  
        {"text": "Budget-constrained optimization", "answer": 2},      # polynomial_regression
        {"text": "Uncertainty quantification needed", "answer": 3},    # gaussian_process
        {"text": "High-dimensional efficient prediction", "answer": 4}, # random_forest
        {"text": "Discontinuous response surface", "answer": 5},       # support_vector_machine
    ]
    
    classification_path = Path("classification_dataset.json")
    with open(classification_path, 'w') as f:
        json.dump(classification_data, f, indent=2)
    
    # Initialize model for classification
    model = TrainableLLM(use_lora=True)
    tokenizer = model.get_tokenizer()
    
    tuner = ComparisonTuner(
        model=model,
        tokenizer=tokenizer,
        output_dir="./classification_tuning_results"
    )
    
    # Compare with classification method
    results = tuner.load_and_compare_dataset(
        dataset_path=classification_path,
        text_column="text",
        answer_column="answer",
        task_type="classification",
        comparison_method="exact_match"
    )
    
    print(f"Classification Results:")
    print(f"Accuracy: {results['overall_metrics']['accuracy']:.3f}")
    print(f"Correct: {results['overall_metrics']['correct_predictions']}")
    print(f"Incorrect: {results['overall_metrics']['incorrect_predictions']}")

if __name__ == "__main__":
    main()
    # demonstrate_classification_tuning()
