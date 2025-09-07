from pathlib import Path
import os
import json
import pandas as pd
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from hypersurrogatemodel import (
    TrainableLLM, 
    ComparisonTuner,
    ConfigManager,
    Logger,
    set_random_seed
)
from hypersurrogatemodel.utils import get_device

logger = Logger("Pipelined-runner")
torch.set_float32_matmul_precision('high')
class log_iterator():
    def __init__(self, lst):
        self.iterable = lst
        self.index = 0
        self.length = len(lst)
        self.current_item = None
    
    def __iter__(self):
        return self
    
    def next(self):
        if self.current_item is not None:
            logger.success(f"Step {self.index}/{self.length}: {self.current_item} success")
        if self.index == self.length:
            return
        if self.index < self.length:
            self.current_item = self.iterable[self.index]
            self.index += 1
            logger.info(f"Step {self.index}/{self.length}: {self.current_item}...")
        else:
            raise StopIteration
        
class DatasetStruct():
    def __init__(self,data_name:str, data:pd.DataFrame):  # input needs: source data, data name, data description, data goal
        self.data_name = data_name
        self.datas = data
        
    def data_cleaning (self):
        self.answer = self.datas[["true_final_train_accuracy"]]
        self.text = self.datas[['uid', 'unified_text_description']]
        

        answer_dict = self.answer.to_dict(orient='records')
        data_dict = self.text.to_dict(orient='records')
        self.wrapped_data = []
        for data, answer in zip(data_dict, answer_dict):
            self.wrapped_data.append({"text": str(data), "answer": str(answer)})  
        logger.success("Data cleaned: reformat to json serializable")
        
    
    def save_to_json(self, file_path:str):
        with open(file_path, 'w+') as f: 
            json.dump(self.wrapped_data, f, indent=2, ensure_ascii=False)
        logger.success(f"DataFrame saved as JSON to {file_path}")
    
def NAS_bench_201():
    logger.step("loading data set")
    original_data_path = Path(os.getcwd()) / "data" / "processed" / "master_dataset.parquet"
    try:
        original_data = pd.read_parquet(original_data_path)
        logger.success(f"data set loaded from {original_data_path}")
        
        dataset_union = []
        for dataset_source in original_data['dataset_source'].unique():
            dataset_union.append(DatasetStruct(data_name=dataset_source, data=original_data[original_data['dataset_source']==dataset_source]))
        
        
    except Exception as e:
        logger.error(str(e))
    
    json_output_path = Path(os.getcwd()) / "data" / "processed" / "NAS_bench_201"
    pathjoint = []
    for ds in dataset_union:
        ds.data_cleaning()
        ds.save_to_json(json_output_path / f"{ds.data_name}_cleaned.json")
        pathjoint.append(json_output_path / f"{ds.data_name}_cleaned.json")
    return pathjoint
    

def tune_with_dataset(dataset_path, model_path = None,):
    
    logger.setFunctionsName("tune_with_dataset")
    steps = ["Load model", "Load tokenizer", "Load optimizer and scheduler", "Initialize tuner"]
    lg = log_iterator(steps)
    lg.next()
    model = TrainableLLM(
        base_model_name = model_path if model_path else "google/gemma-3-270m-it",
        use_lora=True,
        lora_config={
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "target_modules": ["q_proj", "v_proj"],
        }
    )
    logger.info("Model info:")
    model_info = model.get_model_info()
    logger.info(str(model_info))
    lg.next()
    
    optimizer = AdamW(model.parameters(), lr=1e-4)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=10, 
        num_training_steps=100
    )
    lg.next()
    
    tokenizer = model.get_tokenizer()
    lg.next()
    
    tuner = ComparisonTuner(
        model=model,
        tokenizer=tokenizer,
        output_dir="./comparison_tuning_results",
        use_wandb=False,  # weight and bias storage
        save_files=False
    )
    lg.next()
    
    ##### start training and evaluation #####
    logger.step("start evaluation with dataset...")
    
    initial_results = tuner.load_and_compare_dataset(
        dataset_path=dataset_path,
        text_column="text",
        answer_column="answer",
        task_type="generation",
        comparison_method="similarity"
    )
    lg.next()
    
    
    logger.step("start training...")
    checkpoint_dir = "./training_checkpoints"
    
    # Print initial results
    logger.info(f"\n\nInitial Performance:")
    logger.info(f"Accuracy: {initial_results['overall_metrics']['accuracy']:.3f}")
    logger.info(f"Average Similarity: {initial_results['overall_metrics']['average_similarity']:.3f}")
    logger.info(f"Correct Predictions: {initial_results['overall_metrics']['correct_predictions']}")
    logger.info(f"Incorrect Predictions: {initial_results['overall_metrics']['incorrect_predictions']}")
    
    logger.info(f"\n\nError Analysis:")
    error_analysis = initial_results['error_analysis']
    logger.info(f"Error Types: {error_analysis['error_types']}")
    if error_analysis['most_common_error']:
        logger.info(f"Most Common Error: {error_analysis['most_common_error']}")
    
    # Show some examples of differences
    logger.info(f"\\nExample Differences (showing first 3):")
    for i, diff in enumerate(initial_results['differences'][:3]):
        logger.info(f"\\nExample {i+1}:")
        logger.info(f"Input: {diff['input_text'][:100]}...")
        logger.info(f"Predicted: {diff['prediction']}")
        logger.info(f"Expected: {diff['ground_truth']}")
        logger.info(f"Similarity: {diff['similarity']:.3f}")
    
    print("\\n" + "="*60)
    logger.step("STEP 2: Adaptive Tuning Based on Differences")
    print("="*60)
    
    # Perform adaptive tuning if there are errors
    if initial_results['overall_metrics']['incorrect_predictions'] > 0:
        logger.info("Starting adaptive tuning...")
        
        # Try different tuning strategies
        strategies = ["error_focused", "incremental"]
        
        for strategy in strategies:
            logger.info(f"\\n--- Trying {strategy} strategy ---")
            
            tuning_results = tuner.adaptive_tuning(
                comparison_results=initial_results,
                dataset_path=dataset_path,
                text_column="text",
                answer_column="answer",
                tuning_strategy=strategy,
                max_epochs=2,  # Keep it small for demonstration
                learning_rate=1e-5,
            )
            
            logger.info(f"\n\nTuning Results for {strategy}:")
            logger.info(f"Training Data Size: {tuning_results['training_data_size']}")
            
            pre_metrics = tuning_results['pre_tuning_metrics']
            post_metrics = tuning_results['post_tuning_metrics']
            
            logger.info(f"\n\nBefore Tuning:")
            logger.info(f"Accuracy: {pre_metrics['accuracy']:.3f}")
            logger.info(f"Avg Similarity: {pre_metrics['average_similarity']:.3f}")
            
            logger.info(f"\n\nAfter Tuning:")
            logger.info(f"Accuracy: {post_metrics['accuracy']:.3f}")
            logger.info(f"Avg Similarity: {post_metrics['average_similarity']:.3f}")
            
            # Show improvement analysis
            improvements = tuning_results['improvement_analysis']['improvements']
            print(f"\n\nImprovements:")
            for metric, improvement in improvements.items():
                logger.info(f"{metric}: {improvement['absolute_improvement']:+.3f} "
                     f"({improvement['percentage_improvement']:+.1f}%)")
            
            error_reduction = tuning_results['improvement_analysis']['error_reduction']
            logger.info(f"Error Reduction: {error_reduction} samples")
            
            # Break after first successful strategy for demo
            if error_reduction > 0:
                logger.success(f"{strategy} strategy showed improvement!")
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

if __name__ == "__main__":
    file_path = NAS_bench_201()
    tune_with_dataset(dataset_path=file_path[0])
    
    
    
    