"""
Feedback and Evaluation Interface Module

This module provides comprehensive feedback and evaluation capabilities for
the Enhanced LLM Model, including performance m    def evaluate_classification(
        self,
        test_data: List[Dict[str, Any]],
        batch_size: int = 32,
        save_results: bool = True,
        output_dir: Union[str, Path] = "./evaluation_results",
    ) -> Dict[str, Any]:ng, error analysis,
and continuous improvement mechanisms.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict

from .model import TrainableLLM, TextGenerationModel
from .utils import Logger

# Set up logger using utils.Logger
logger = Logger("evaluator")


class PerformanceMonitor:
    """
    Monitors and tracks model performance over time.
    """
    
    def __init__(self, save_dir: str = "./performance_logs"):
        """
        Initialize the performance monitor.
        
        Args:
            save_dir: Directory to save performance logs
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.performance_history = []
    
    def log_performance(
        self,
        metrics: Dict[str, float],
        epoch: int,
        dataset_name: str = "eval",
        model_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log performance metrics for a specific epoch.
        
        Args:
            metrics: Dictionary of performance metrics
            epoch: Training epoch number
            dataset_name: Name of the dataset (e.g., "train", "eval", "test")
            model_info: Additional model information
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            "dataset": dataset_name,
            "metrics": metrics,
            "model_info": model_info or {},
        }
        
        self.performance_history.append(log_entry)
        
        # Save to file
        log_file = self.save_dir / f"performance_log_{datetime.now().strftime('%Y%m%d')}.json"
        with open(log_file, 'w') as f:
            json.dump(self.performance_history, f, indent=2, default=str)
        
        logger.info(f"Performance logged for epoch {epoch}: {metrics}")
    
    def get_performance_trend(
        self,
        metric: str = "accuracy",
        dataset: str = "eval",
    ) -> List[Tuple[int, float]]:
        """
        Get performance trend for a specific metric.
        
        Args:
            metric: Metric name to track
            dataset: Dataset name to filter by
            
        Returns:
            List of (epoch, metric_value) tuples
        """
        trend = []
        for entry in self.performance_history:
            if entry["dataset"] == dataset and metric in entry["metrics"]:
                trend.append((entry["epoch"], entry["metrics"][metric]))
        
        return sorted(trend, key=lambda x: x[0])
    
    def plot_performance_trends(
        self,
        metrics: List[str] = ["accuracy", "f1"],
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot performance trends over epochs.
        
        Args:
            metrics: List of metrics to plot
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 8))
        
        for i, metric in enumerate(metrics):
            plt.subplot(2, 2, i + 1)
            
            for dataset in ["train", "eval", "test"]:
                trend = self.get_performance_trend(metric, dataset)
                if trend:
                    epochs, values = zip(*trend)
                    plt.plot(epochs, values, label=f"{dataset}_{metric}", marker='o')
            
            plt.title(f"{metric.capitalize()} Over Time")
            plt.xlabel("Epoch")
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.save_dir / "performance_trends.png", dpi=300, bbox_inches='tight')
        
        plt.show()


class ModelEvaluator:
    """
    Comprehensive model evaluation with detailed analysis.
    """
    
    def __init__(
        self,
        model: Union[TrainableLLM, TextGenerationModel],
        tokenizer,
        class_names: Optional[List[str]] = None,
    ):
        """
        Initialize the evaluator.
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer for text processing
            class_names: Names of classes for classification tasks
        """
        self.model = model
        self.tokenizer = tokenizer
        self.class_names = class_names or [f"Class_{i}" for i in range(12)]
        
        # Set model to evaluation mode
        self.model.eval()
    
    def evaluate_classification(
        self,
        test_data: List[Dict[str, Any]],
        batch_size: int = 8,
        save_results: bool = True,
        output_dir: Union[str, Path] = "./evaluation_results",
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation for classification tasks.
        
        Args:
            test_data: List of test samples with "text" and "label" keys
            batch_size: Batch size for evaluation
            save_results: Whether to save detailed results
            output_dir: Directory to save results
            
        Returns:
            Comprehensive evaluation results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        predictions = []
        true_labels = []
        prediction_probabilities = []
        sample_texts = []
        
        logger.info("Starting classification evaluation...")
        
        # Process in batches
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i + batch_size]
            batch_texts = [item["text"] for item in batch]
            batch_labels = [item["label"] for item in batch]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
                
                logits = outputs["logits"]
                probabilities = torch.softmax(logits, dim=-1)
                batch_predictions = torch.argmax(logits, dim=-1)
                
                predictions.extend(batch_predictions.cpu().numpy())
                true_labels.extend(batch_labels)
                prediction_probabilities.extend(probabilities.cpu().numpy())
                sample_texts.extend(batch_texts)
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        prediction_probabilities = np.array(prediction_probabilities)
        
        # Basic metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        
        # Per-class metrics
        per_class_metrics = precision_recall_fscore_support(
            true_labels, predictions, average=None, labels=range(len(self.class_names))
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # Classification report
        report = classification_report(
            true_labels, predictions,
            target_names=self.class_names,
            output_dict=True
        )
        
        # Error analysis
        error_analysis = self._analyze_classification_errors(
            sample_texts, true_labels, predictions, prediction_probabilities
        )
        
        # Compile results
        results = {
            "basic_metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            },
            "per_class_metrics": {
                "precision": per_class_metrics[0].tolist(), # type: ignore
                "recall": per_class_metrics[1].tolist(), # type: ignore
                "f1": per_class_metrics[2].tolist(), # type: ignore
                "support": per_class_metrics[3].tolist(), # type: ignore
            },
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "error_analysis": error_analysis,
            "sample_predictions": self._get_sample_predictions(
                sample_texts, true_labels, predictions, prediction_probabilities
            ),
        }
        
        if save_results:
            # Save results
            with open(output_path / "evaluation_results.json", 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Save visualizations
            self._plot_confusion_matrix(cm, output_path / "confusion_matrix.png")
            self._plot_per_class_metrics(per_class_metrics, output_path / "per_class_metrics.png")
            
            logger.info(f"Evaluation results saved to {output_path}")
        
        return results
    
    def _analyze_classification_errors(
        self,
        texts: List[str],
        true_labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Analyze classification errors to identify patterns.
        
        Args:
            texts: Input texts
            true_labels: True labels
            predictions: Predicted labels
            probabilities: Prediction probabilities
            
        Returns:
            Error analysis results
        """
        # Find misclassified samples
        errors = true_labels != predictions
        error_indices = np.where(errors)[0]
        
        error_samples = []
        confidence_errors = []
        pattern_analysis = defaultdict(int)
        
        for idx in error_indices:
            confidence = probabilities[idx][predictions[idx]]
            
            error_sample = {
                "text": texts[idx],
                "true_label": int(true_labels[idx]),
                "predicted_label": int(predictions[idx]),
                "confidence": float(confidence),
                "true_class_name": self.class_names[true_labels[idx]],
                "predicted_class_name": self.class_names[predictions[idx]],
            }
            error_samples.append(error_sample)
            confidence_errors.append(confidence)
            
            # Pattern analysis
            pattern_key = f"{true_labels[idx]}->{predictions[idx]}"
            pattern_analysis[pattern_key] += 1
        
        # Most common error patterns
        common_errors = sorted(pattern_analysis.items(), key=lambda x: x[1], reverse=True)
        
        # Confidence analysis
        avg_error_confidence = np.mean(confidence_errors) if confidence_errors else 0
        
        return {
            "total_errors": len(error_samples),
            "error_rate": len(error_samples) / len(texts),
            "average_error_confidence": avg_error_confidence,
            "common_error_patterns": common_errors[:10],
            "error_samples": error_samples[:20],  # Top 20 errors
        }
    
    def _get_sample_predictions(
        self,
        texts: List[str],
        true_labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        num_samples: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get sample predictions for manual inspection."""
        samples = []
        indices = np.random.choice(len(texts), min(num_samples, len(texts)), replace=False)
        
        for idx in indices:
            samples.append({
                "text": texts[idx],
                "true_label": int(true_labels[idx]),
                "predicted_label": int(predictions[idx]),
                "prediction_probabilities": probabilities[idx].tolist(),
                "is_correct": bool(true_labels[idx] == predictions[idx]),
            })
        
        return samples
    
    def _plot_confusion_matrix(self, cm: np.ndarray, save_path: Path) -> None:
        """Plot and save confusion matrix."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_per_class_metrics(self, metrics: Tuple, save_path: Path) -> None:
        """Plot per-class metrics."""
        precision, recall, f1, support = metrics
        
        x = np.arange(len(self.class_names))
        width = 0.25
        
        plt.figure(figsize=(12, 6))
        plt.bar(x - width, precision, width, label='Precision', alpha=0.8)
        plt.bar(x, recall, width, label='Recall', alpha=0.8)
        plt.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        plt.xlabel('Classes')
        plt.ylabel('Score')
        plt.title('Per-Class Performance Metrics')
        plt.xticks(x, self.class_names, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def evaluate_generation(
        self,
        test_prompts: List[str],
        reference_outputs: Optional[List[str]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        save_results: bool = True,
        output_dir: Union[str, Path] = "./generation_evaluation",
    ) -> Dict[str, Any]:
        """
        Evaluate text generation capabilities.
        
        Args:
            test_prompts: List of input prompts
            reference_outputs: Optional reference outputs for comparison
            generation_kwargs: Parameters for text generation
            save_results: Whether to save results
            output_dir: Directory to save results
            
        Returns:
            Generation evaluation results
        """
        if not isinstance(self.model, TextGenerationModel):
            raise ValueError("Generation evaluation requires TextGenerationModel")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        generation_kwargs = generation_kwargs or {
            "max_new_tokens": 100,
            "temperature": 0.7,
            "do_sample": True,
        }
        
        generated_outputs = []
        generation_metrics = []
        
        logger.info("Starting generation evaluation...")
        
        for i, prompt in enumerate(test_prompts):
            logger.info(f"Generating for prompt {i+1}/{len(test_prompts)}")
            
            # Generate text
            generated_text = self.model.generate_text(prompt, **generation_kwargs)
            generated_outputs.append(generated_text)
            
            # Calculate basic metrics
            metrics = {
                "prompt_length": len(prompt),
                "generated_length": len(generated_text),
                "tokens_generated": len(self.tokenizer.encode(generated_text)),
            }
            
            if reference_outputs and i < len(reference_outputs):
                reference = reference_outputs[i]
                # Calculate similarity metrics (simplified)
                additional_metrics = {
                    "reference_length": len(reference),
                    "length_ratio": len(generated_text) / len(reference) if reference else 0,
                }
                metrics.update(additional_metrics)
            
            generation_metrics.append(metrics)
        
        # Aggregate metrics
        avg_metrics = {
            "avg_prompt_length": np.mean([m["prompt_length"] for m in generation_metrics]),
            "avg_generated_length": np.mean([m["generated_length"] for m in generation_metrics]),
            "avg_tokens_generated": np.mean([m["tokens_generated"] for m in generation_metrics]),
        }
        
        if reference_outputs:
            avg_metrics["avg_length_ratio"] = np.mean([
                m.get("length_ratio", 0) for m in generation_metrics
            ])
        
        # Quality analysis
        quality_analysis = self._analyze_generation_quality(
            test_prompts, generated_outputs, reference_outputs
        )
        
        results = {
            "prompts": test_prompts,
            "generated_outputs": generated_outputs,
            "reference_outputs": reference_outputs,
            "generation_metrics": generation_metrics,
            "average_metrics": avg_metrics,
            "quality_analysis": quality_analysis,
        }
        
        if save_results:
            with open(output_path / "generation_results.json", 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            # Save human-readable format
            self._save_generation_samples(results, output_path / "generation_samples.txt")
            
            logger.info(f"Generation evaluation results saved to {output_path}")
        
        return results
    
    def _analyze_generation_quality(
        self,
        prompts: List[str],
        outputs: List[str],
        references: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze the quality of generated text.
        
        Args:
            prompts: Input prompts
            outputs: Generated outputs
            references: Reference outputs (optional)
            
        Returns:
            Quality analysis results
        """
        analysis = {
            "repetition_analysis": self._analyze_repetition(outputs),
            "diversity_metrics": self._calculate_diversity_metrics(outputs),
            "coherence_indicators": self._analyze_coherence(prompts, outputs),
        }
        
        if references:
            analysis["similarity_metrics"] = self._calculate_similarity_metrics(outputs, references)
        
        return analysis
    
    def _analyze_repetition(self, outputs: List[str]) -> Dict[str, float]:
        """Analyze repetition in generated texts."""
        repetition_scores = []
        
        for text in outputs:
            words = text.split()
            if len(words) <= 1:
                repetition_scores.append(0.0)
                continue
            
            unique_words = len(set(words))
            total_words = len(words)
            repetition_ratio = 1 - (unique_words / total_words)
            repetition_scores.append(repetition_ratio)
        
        return {
            "avg_repetition_ratio": float(np.mean(repetition_scores)),
            "max_repetition_ratio": float(np.max(repetition_scores)),
            "min_repetition_ratio": float(np.min(repetition_scores)),
        }
    
    def _calculate_diversity_metrics(self, outputs: List[str]) -> Dict[str, float]:
        """Calculate diversity metrics across outputs."""
        all_words = []
        for text in outputs:
            all_words.extend(text.split())
        
        if not all_words:
            return {"vocabulary_diversity": 0.0}
        
        unique_words = len(set(all_words))
        total_words = len(all_words)
        
        return {
            "vocabulary_diversity": unique_words / total_words if total_words > 0 else 0,
            "total_unique_words": unique_words,
            "total_words": total_words,
        }
    
    def _analyze_coherence(self, prompts: List[str], outputs: List[str]) -> Dict[str, Any]:
        """Analyze coherence between prompts and outputs."""
        coherence_indicators = []
        
        for prompt, output in zip(prompts, outputs):
            # Simple coherence check: keyword overlap
            prompt_words = set(prompt.lower().split())
            output_words = set(output.lower().split())
            
            overlap = len(prompt_words & output_words)
            coherence_score = overlap / len(prompt_words) if prompt_words else 0
            
            coherence_indicators.append({
                "keyword_overlap": overlap,
                "coherence_score": coherence_score,
                "output_starts_with_context": output.lower().startswith(prompt.lower()[:20]),
            })
        
        return {
            "avg_coherence_score": np.mean([c["coherence_score"] for c in coherence_indicators]),
            "avg_keyword_overlap": np.mean([c["keyword_overlap"] for c in coherence_indicators]),
            "context_continuation_rate": np.mean([
                c["output_starts_with_context"] for c in coherence_indicators
            ]),
        }
    
    def _calculate_similarity_metrics(
        self, outputs: List[str], references: List[str]
    ) -> Dict[str, float]:
        """Calculate similarity between outputs and references."""
        # This is a simplified implementation
        # In practice, you might want to use more sophisticated metrics like BLEU, ROUGE, etc.
        
        similarities = []
        for output, reference in zip(outputs, references):
            # Simple word-based similarity
            output_words = set(output.lower().split())
            reference_words = set(reference.lower().split())
            
            if not reference_words:
                similarities.append(0.0)
                continue
            
            intersection = len(output_words & reference_words)
            union = len(output_words | reference_words)
            
            jaccard_similarity = intersection / union if union > 0 else 0
            similarities.append(jaccard_similarity)
        
        return {
            "avg_jaccard_similarity": float(np.mean(similarities)),
            "max_similarity": float(np.max(similarities)),
            "min_similarity": float(np.min(similarities)),
        }
    
    def _save_generation_samples(self, results: Dict[str, Any], save_path: Path) -> None:
        """Save generation samples in human-readable format."""
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=== Text Generation Evaluation Samples ===\n\n")
            
            for i, (prompt, output) in enumerate(zip(
                results["prompts"], results["generated_outputs"]
            )):
                f.write(f"Sample {i+1}:\n")
                f.write(f"Prompt: {prompt}\n")
                f.write(f"Generated: {output}\n")
                
                if results["reference_outputs"] and i < len(results["reference_outputs"]):
                    f.write(f"Reference: {results['reference_outputs'][i]}\n")
                
                f.write("-" * 80 + "\n\n")


class FeedbackCollector:
    """
    Collects and manages human feedback for model improvement.
    """
    
    def __init__(self, feedback_dir: str = "./feedback"):
        """
        Initialize feedback collector.
        
        Args:
            feedback_dir: Directory to store feedback data
        """
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_data = []
    
    def collect_classification_feedback(
        self,
        text: str,
        predicted_label: int,
        correct_label: Optional[int] = None,
        confidence: Optional[float] = None,
        user_id: Optional[str] = None,
        comments: Optional[str] = None,
    ) -> str:
        """
        Collect feedback for classification predictions.
        
        Args:
            text: Input text
            predicted_label: Model's predicted label
            correct_label: Correct label (if known)
            confidence: Model's confidence in prediction
            user_id: ID of the user providing feedback
            comments: Additional comments
            
        Returns:
            Feedback ID
        """
        feedback_id = f"clf_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.feedback_data)}"
        
        feedback_entry = {
            "feedback_id": feedback_id,
            "timestamp": datetime.now().isoformat(),
            "type": "classification",
            "text": text,
            "predicted_label": predicted_label,
            "correct_label": correct_label,
            "confidence": confidence,
            "user_id": user_id,
            "comments": comments,
        }
        
        self.feedback_data.append(feedback_entry)
        self._save_feedback()
        
        logger.info(f"Classification feedback collected: {feedback_id}")
        return feedback_id
    
    def collect_generation_feedback(
        self,
        prompt: str,
        generated_text: str,
        quality_rating: Optional[int] = None,  # 1-5 scale
        relevance_rating: Optional[int] = None,  # 1-5 scale
        improved_text: Optional[str] = None,
        user_id: Optional[str] = None,
        comments: Optional[str] = None,
    ) -> str:
        """
        Collect feedback for text generation.
        
        Args:
            prompt: Input prompt
            generated_text: Generated text
            quality_rating: Quality rating (1-5)
            relevance_rating: Relevance rating (1-5)
            improved_text: User's improved version
            user_id: ID of the user providing feedback
            comments: Additional comments
            
        Returns:
            Feedback ID
        """
        feedback_id = f"gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.feedback_data)}"
        
        feedback_entry = {
            "feedback_id": feedback_id,
            "timestamp": datetime.now().isoformat(),
            "type": "generation",
            "prompt": prompt,
            "generated_text": generated_text,
            "quality_rating": quality_rating,
            "relevance_rating": relevance_rating,
            "improved_text": improved_text,
            "user_id": user_id,
            "comments": comments,
        }
        
        self.feedback_data.append(feedback_entry)
        self._save_feedback()
        
        logger.info(f"Generation feedback collected: {feedback_id}")
        return feedback_id
    
    def _save_feedback(self) -> None:
        """Save feedback data to file."""
        feedback_file = self.feedback_dir / "feedback_data.json"
        with open(feedback_file, 'w') as f:
            json.dump(self.feedback_data, f, indent=2, default=str)
    
    def load_feedback(self) -> List[Dict[str, Any]]:
        """Load existing feedback data."""
        feedback_file = self.feedback_dir / "feedback_data.json"
        if feedback_file.exists():
            with open(feedback_file, 'r') as f:
                self.feedback_data = json.load(f)
        return self.feedback_data
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get summary of collected feedback."""
        if not self.feedback_data:
            return {"total_feedback": 0}
        
        feedback_by_type = defaultdict(int)
        ratings_summary = {"quality": [], "relevance": []}
        
        for feedback in self.feedback_data:
            feedback_by_type[feedback["type"]] += 1
            
            if feedback["type"] == "generation":
                if feedback.get("quality_rating"):
                    ratings_summary["quality"].append(feedback["quality_rating"])
                if feedback.get("relevance_rating"):
                    ratings_summary["relevance"].append(feedback["relevance_rating"])
        
        summary = {
            "total_feedback": len(self.feedback_data),
            "feedback_by_type": dict(feedback_by_type),
            "average_quality_rating": np.mean(ratings_summary["quality"]) if ratings_summary["quality"] else None,
            "average_relevance_rating": np.mean(ratings_summary["relevance"]) if ratings_summary["relevance"] else None,
        }
        
        return summary
    
    def export_feedback_for_training(
        self,
        output_path: str,
        feedback_type: Optional[str] = None,
        min_quality_rating: Optional[int] = None,
    ) -> None:
        """
        Export feedback data in format suitable for retraining.
        
        Args:
            output_path: Path to save the exported data
            feedback_type: Filter by feedback type
            min_quality_rating: Minimum quality rating to include
        """
        filtered_data = []
        
        for feedback in self.feedback_data:
            # Apply filters
            if feedback_type and feedback["type"] != feedback_type:
                continue
            
            if (min_quality_rating and 
                feedback.get("quality_rating", 0) < min_quality_rating):
                continue
            
            if feedback["type"] == "classification" and feedback.get("correct_label") is not None:
                filtered_data.append({
                    "text": feedback["text"],
                    "label": feedback["correct_label"],
                })
            elif feedback["type"] == "generation" and feedback.get("improved_text"):
                filtered_data.append({
                    "input": feedback["prompt"],
                    "output": feedback["improved_text"],
                })
        
        with open(output_path, 'w') as f:
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(filtered_data)} feedback samples to {output_path}")


class ModelDiagnostics:
    """
    Diagnostic tools for model analysis and debugging.
    """
    
    def __init__(self, model: Union[TrainableLLM, TextGenerationModel]):
        """
        Initialize diagnostics.
        
        Args:
            model: Model to analyze
        """
        self.model = model
    
    def analyze_attention_patterns(
        self,
        input_text: str,
        layer_idx: int = -1,
        head_idx: int = 0,
    ) -> Dict[str, Any]:
        """
        Analyze attention patterns for interpretability.
        
        Args:
            input_text: Input text to analyze
            layer_idx: Layer index to analyze
            head_idx: Attention head index
            
        Returns:
            Attention analysis results
        """
        # This would require model modifications to output attention weights
        # Simplified implementation
        logger.info("Attention pattern analysis would require model modifications")
        return {"message": "Attention analysis not implemented in this simplified version"}
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive model statistics.
        
        Returns:
            Model statistics
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Memory usage estimation
        model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
        
        stats = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": total_params - trainable_params,
            "trainable_ratio": trainable_params / total_params * 100,
            "estimated_model_size_mb": model_size_mb,
        }
        
        # Add model-specific information
        try:
            if hasattr(self.model, 'get_model_info') and callable(getattr(self.model, 'get_model_info')):
                model_info = getattr(self.model, 'get_model_info')()
                if isinstance(model_info, dict):
                    stats.update(model_info)
        except Exception:
            # Ignore if model info cannot be retrieved
            pass
        
        return stats
