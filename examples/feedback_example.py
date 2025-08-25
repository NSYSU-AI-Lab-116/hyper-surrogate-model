"""
Feedback Interface Example

This example demonstrates how to use the feedback and evaluation interface
for continuous model improvement.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hypersurrogatemodel import (
    EnhancedLLMModel,
    ModelEvaluator,
    PerformanceMonitor,
    FeedbackCollector,
    ModelDiagnostics,
    Logger
)
import torch
import json
from pathlib import Path


def main():
    """
    Main function demonstrating the feedback interface workflow.
    """
    # Initialize logger
    logger = Logger("feedback_example")
    logger.info("Starting feedback interface example")
    
    # Step 1: Load a trained model (or use a fresh one for demo)
    logger.info("Initializing model...")
    model = EnhancedLLMModel(
        base_model_name="google/gemma-3-270m-it",
        num_classes=12,
        hidden_size=256,
        dropout_rate=0.1,
        use_lora=True,
    )
    
    tokenizer = model.get_tokenizer()
    
    # Step 2: Initialize feedback and evaluation components
    logger.info("Setting up feedback and evaluation interfaces...")
    
    # Performance monitor for tracking metrics over time
    performance_monitor = PerformanceMonitor(save_dir="./performance_logs")
    
    # Model evaluator for comprehensive evaluation
    class_names = [
        "Technology", "Healthcare", "Finance", "Education",
        "Entertainment", "Sports", "Politics", "Science",
        "Travel", "Food", "Fashion", "Environment"
    ]
    evaluator = ModelEvaluator(
        model=model,
        tokenizer=tokenizer,
        class_names=class_names
    )
    
    # Feedback collector for gathering user feedback
    feedback_collector = FeedbackCollector(feedback_dir="./feedback")
    
    # Model diagnostics for analysis
    diagnostics = ModelDiagnostics(model=model)
    
    # Step 3: Create sample test data for evaluation
    logger.info("Creating test data...")
    test_data = create_test_data()
    
    # Step 4: Perform comprehensive evaluation
    logger.info("Performing model evaluation...")
    eval_results = evaluator.evaluate_classification(
        test_data=test_data,
        batch_size=4,
        save_results=True,
        output_dir="./evaluation_results"
    )
    
    # Log performance metrics
    performance_monitor.log_performance(
        metrics=eval_results["basic_metrics"],
        epoch=1,
        dataset_name="test",
        model_info=model.get_model_info()
    )
    
    # Step 5: Demonstrate feedback collection
    logger.info("Demonstrating feedback collection...")
    demonstrate_feedback_collection(feedback_collector, test_data[:5])
    
    # Step 6: Analyze model diagnostics
    logger.info("Analyzing model diagnostics...")
    model_stats = diagnostics.get_model_statistics()
    print("\n" + "="*50)
    print("MODEL DIAGNOSTICS")
    print("="*50)
    for key, value in model_stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # Step 7: Generate feedback summary and recommendations
    logger.info("Generating feedback summary...")
    feedback_summary = feedback_collector.get_feedback_summary()
    
    print("\n" + "="*50)
    print("FEEDBACK SUMMARY")
    print("="*50)
    print(json.dumps(feedback_summary, indent=2))
    
    # Step 8: Export feedback for potential retraining
    logger.info("Exporting feedback for retraining...")
    feedback_collector.export_feedback_for_training(
        output_path="./feedback_for_training.json",
        min_quality_rating=3
    )
    
    # Step 9: Performance visualization
    logger.info("Creating performance visualizations...")
    try:
        # Add a few more performance logs for demonstration
        for epoch in range(2, 6):
            # Simulate improving performance
            simulated_metrics = {
                "accuracy": 0.7 + epoch * 0.05,
                "f1": 0.65 + epoch * 0.06,
                "precision": 0.68 + epoch * 0.04,
                "recall": 0.66 + epoch * 0.05,
            }
            performance_monitor.log_performance(
                metrics=simulated_metrics,
                epoch=epoch,
                dataset_name="test"
            )
        
        # Plot performance trends
        performance_monitor.plot_performance_trends(
            metrics=["accuracy", "f1"],
            save_path="./performance_trends.png"
        )
        
        logger.info("Performance trends plotted and saved")
    except Exception as e:
        logger.warning(f"Could not create performance plots: {e}")
    
    # Step 10: Provide improvement recommendations
    logger.info("Generating improvement recommendations...")
    recommendations = generate_improvement_recommendations(
        eval_results, feedback_summary, model_stats
    )
    
    print("\n" + "="*50)
    print("IMPROVEMENT RECOMMENDATIONS")
    print("="*50)
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    logger.info("Feedback interface example completed successfully!")
    
    return {
        "eval_results": eval_results,
        "feedback_summary": feedback_summary,
        "model_stats": model_stats,
        "recommendations": recommendations,
    }


def create_test_data():
    """
    Create sample test data for evaluation.
    
    Returns:
        List of test samples with text and label
    """
    test_samples = [
        # Technology samples
        {"text": "The new AI model outperforms previous benchmarks.", "label": 0},
        {"text": "Quantum computing will revolutionize cryptography.", "label": 0},
        {"text": "5G networks enable faster mobile connectivity.", "label": 0},
        
        # Healthcare samples
        {"text": "The vaccine shows 95% efficacy in clinical trials.", "label": 1},
        {"text": "Personalized medicine targets individual genetic profiles.", "label": 1},
        {"text": "Wearable devices monitor patient vital signs continuously.", "label": 1},
        
        # Finance samples
        {"text": "Bitcoin price volatility affects investment strategies.", "label": 2},
        {"text": "Central banks consider digital currency implementation.", "label": 2},
        {"text": "Algorithmic trading dominates modern financial markets.", "label": 2},
        
        # Education samples
        {"text": "Remote learning platforms expand educational access.", "label": 3},
        {"text": "Gamification improves student engagement significantly.", "label": 3},
        {"text": "Adaptive learning systems personalize educational content.", "label": 3},
        
        # Entertainment samples
        {"text": "Streaming platforms compete for exclusive content rights.", "label": 4},
        {"text": "Virtual reality gaming creates immersive experiences.", "label": 4},
        {"text": "Independent filmmakers use crowdfunding for projects.", "label": 4},
        
        # Sports samples
        {"text": "Advanced analytics optimize athlete performance training.", "label": 5},
        {"text": "E-sports tournaments attract millions of viewers worldwide.", "label": 5},
        {"text": "Sports medicine prevents and treats athletic injuries.", "label": 5},
        
        # Politics samples
        {"text": "Electoral reforms aim to increase voter participation.", "label": 6},
        {"text": "International diplomacy addresses global challenges.", "label": 6},
        {"text": "Policy makers debate climate legislation priorities.", "label": 6},
        
        # Science samples
        {"text": "Researchers discover new exoplanets in distant galaxies.", "label": 7},
        {"text": "Gene editing technology treats hereditary diseases.", "label": 7},
        {"text": "Climate models predict future environmental changes.", "label": 7},
        
        # Travel samples
        {"text": "Sustainable tourism preserves natural environments.", "label": 8},
        {"text": "Digital nomads work remotely from exotic locations.", "label": 8},
        {"text": "Cultural immersion programs enrich travel experiences.", "label": 8},
        
        # Food samples
        {"text": "Plant-based proteins offer sustainable nutrition alternatives.", "label": 9},
        {"text": "Molecular gastronomy transforms traditional cooking methods.", "label": 9},
        {"text": "Local food movements support community agriculture.", "label": 9},
        
        # Fashion samples
        {"text": "Sustainable fashion brands use eco-friendly materials.", "label": 10},
        {"text": "3D printing technology creates customized clothing.", "label": 10},
        {"text": "Fashion weeks showcase emerging designer talent.", "label": 10},
        
        # Environment samples
        {"text": "Renewable energy sources reduce carbon emissions.", "label": 11},
        {"text": "Ocean conservation efforts protect marine ecosystems.", "label": 11},
        {"text": "Reforestation projects combat climate change effects.", "label": 11},
    ]
    
    return test_samples


def demonstrate_feedback_collection(feedback_collector, test_samples):
    """
    Demonstrate how to collect different types of feedback.
    
    Args:
        feedback_collector: FeedbackCollector instance
        test_samples: Sample data for feedback collection
    """
    print("\n" + "="*50)
    print("FEEDBACK COLLECTION DEMONSTRATION")
    print("="*50)
    
    # Simulate classification feedback
    for i, sample in enumerate(test_samples):
        predicted_label = (sample["label"] + 1) % 12  # Simulate incorrect prediction
        confidence = 0.75 + (i * 0.05)  # Simulate varying confidence
        
        feedback_id = feedback_collector.collect_classification_feedback(
            text=sample["text"],
            predicted_label=predicted_label,
            correct_label=sample["label"],
            confidence=confidence,
            user_id=f"user_{i}",
            comments=f"Sample feedback for test case {i}"
        )
        
        print(f"Collected classification feedback: {feedback_id}")
    
    # Simulate generation feedback
    generation_examples = [
        {
            "prompt": "Explain quantum computing",
            "generated": "Quantum computing uses quantum bits to process information faster than classical computers.",
            "quality": 4,
            "relevance": 5,
            "improved": "Quantum computing leverages quantum mechanical phenomena like superposition and entanglement to perform calculations exponentially faster than classical computers for specific problems."
        },
        {
            "prompt": "What is machine learning?",
            "generated": "Machine learning is a type of AI.",
            "quality": 2,
            "relevance": 3,
            "improved": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for each task."
        }
    ]
    
    for example in generation_examples:
        feedback_id = feedback_collector.collect_generation_feedback(
            prompt=example["prompt"],
            generated_text=example["generated"],
            quality_rating=example["quality"],
            relevance_rating=example["relevance"],
            improved_text=example["improved"],
            user_id="expert_evaluator",
            comments="Professional evaluation"
        )
        
        print(f"Collected generation feedback: {feedback_id}")


def generate_improvement_recommendations(eval_results, feedback_summary, model_stats):
    """
    Generate actionable recommendations based on evaluation and feedback.
    
    Args:
        eval_results: Model evaluation results
        feedback_summary: Feedback summary statistics
        model_stats: Model diagnostic statistics
        
    Returns:
        List of improvement recommendations
    """
    recommendations = []
    
    # Performance-based recommendations
    accuracy = eval_results["basic_metrics"]["accuracy"]
    f1_score = eval_results["basic_metrics"]["f1"]
    
    if accuracy < 0.8:
        recommendations.append(
            f"Model accuracy ({accuracy:.3f}) is below 0.8. Consider collecting more training data "
            "or adjusting hyperparameters."
        )
    
    if f1_score < 0.75:
        recommendations.append(
            f"F1 score ({f1_score:.3f}) suggests class imbalance issues. "
            "Consider data augmentation or class weighting."
        )
    
    # Error analysis recommendations
    if "error_analysis" in eval_results:
        error_rate = eval_results["error_analysis"]["error_rate"]
        if error_rate > 0.3:
            recommendations.append(
                f"High error rate ({error_rate:.3f}) detected. "
                "Review common error patterns and improve prompt engineering."
            )
    
    # Feedback-based recommendations
    if feedback_summary.get("total_feedback", 0) > 0:
        avg_quality = feedback_summary.get("average_quality_rating")
        if avg_quality and avg_quality < 3.5:
            recommendations.append(
                f"Average user quality rating ({avg_quality:.2f}) is low. "
                "Consider retraining with user-corrected examples."
            )
    
    # Model architecture recommendations
    trainable_ratio = model_stats.get("trainable_ratio", 100)
    if trainable_ratio < 5:
        recommendations.append(
            f"Very low trainable parameter ratio ({trainable_ratio:.2f}%). "
            "Consider increasing LoRA rank or unfreezing more layers."
        )
    
    # General recommendations
    recommendations.extend([
        "Implement regular evaluation cycles to monitor performance drift.",
        "Collect diverse feedback from different user groups for robust evaluation.",
        "Consider A/B testing different model versions before deployment.",
        "Set up automated monitoring for production deployment.",
        "Create a feedback loop for continuous model improvement."
    ])
    
    return recommendations


if __name__ == "__main__":
    try:
        results = main()
        print(f"\nFeedback interface example completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
