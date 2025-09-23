import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
import matplotlib.pyplot as plt
from hypersurrogatemodel import Logger

logger = Logger("ModelEvaluator")

def calculate_metrics(predictions_path: str):
    logger.info(f"Loading predictions from {predictions_path}...")
    try:
        with open(predictions_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Error: File not found at '{predictions_path}'. Please check the file path.")
        return

    true_values = np.array([item['true_answer'] for item in data])
    predicted_values = np.array([item['predicted_score'] for item in data])

    logger.info("Calculation complete. Here are the model evaluation results:")
    print("-" * 60)

    spearman_corr, _ = spearmanr(true_values, predicted_values)
    logger.step(f"Spearman's Rank Correlation (Rho): {spearman_corr:.4f}")
    print("   (A key metric for surrogate models. Closer to 1.0 means better ranking ability.)")
    print("-" * 60)

    kendall_tau, _ = kendalltau(true_values, predicted_values)
    logger.step(f"Kendall's Rank Correlation (Tau): {kendall_tau:.4f}")
    print("   (Another strong indicator of ranking quality. Closer to 1.0 is better.)")
    print("-" * 60)

    logger.step("Top-k Set Overlap Analysis:")
    true_ranking_indices = np.argsort(true_values)[::-1]
    predicted_ranking_indices = np.argsort(predicted_values)[::-1]
    
    top_k_results = {}
    for k in [10, 50, 100, 500]:
        true_top_k = set(true_ranking_indices[:k])
        predicted_top_k = set(predicted_ranking_indices[:k])
        overlap = len(true_top_k.intersection(predicted_top_k))
        overlap_percentage = (overlap / k) * 100
        top_k_results[f'Top-{k} Overlap (%)'] = overlap_percentage
        
        print(f"  - Top-{k}:")
        print(f"    Found {overlap} common architectures in the top {k} sets.")
        print(f"    Overlap Percentage: {overlap_percentage:.2f}%")
        
    print("  (Measures how accurately the model identifies elite architectures.)")
    print("-" * 60)
    
    # --- Markdown Table Export Section ---
    results_data = {
        "Metric": [
            "Spearman's Rho",
            "Kendall's Tau",
            "Top-10 Overlap (%)",
            "Top-50 Overlap (%)",
            "Top-100 Overlap (%)",
            "Top-500 Overlap (%)"
        ],
        "Value": [
            f"{spearman_corr:.4f}",
            f"{kendall_tau:.4f}",
            f"{top_k_results['Top-10 Overlap (%)']:.2f}",
            f"{top_k_results['Top-50 Overlap (%)']:.2f}",
            f"{top_k_results['Top-100 Overlap (%)']:.2f}",
            f"{top_k_results['Top-500 Overlap (%)']:.2f}"
        ]
    }
    df = pd.DataFrame(results_data)
    
    # Convert the DataFrame to a Markdown string
    markdown_table = df.to_markdown(index=False)
    
    # Write the string to a .md file
    table_path = "evaluation_summary.md"
    with open(table_path, 'w') as f:
        f.write("# Model Evaluation Summary\n\n")
        f.write(markdown_table)
        
    logger.success(f"Evaluation summary table saved to {table_path}")
    print("-" * 60)
    
    # --- Visualization Section ---
    plt.figure(figsize=(10, 8))
    plt.scatter(true_values, predicted_values, alpha=0.3, label=f"Spearman's ρ = {spearman_corr:.4f}")
    
    plt.title("True Accuracy vs. Predicted Ranking Score")
    plt.xlabel("True Accuracy")
    plt.ylabel("Predicted Ranking Score (Unitless)") 
    plt.grid(True)
    plt.legend()
    
    plot_path = "evaluation_scatter_plot.png"
    plt.savefig(plot_path)
    logger.success(f"Evaluation scatter plot saved to {plot_path}")


if __name__ == "__main__":
    PREDICTIONS_FILE_PATH = "./data/results/predictions.json"
    calculate_metrics(PREDICTIONS_FILE_PATH)