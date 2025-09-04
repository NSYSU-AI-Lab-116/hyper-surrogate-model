"""
Quick Start Guide: Dataset Comparison and Tuning

This guide shows the simplest way to use the comparison tuning functionality.
"""

import json
from pathlib import Path
from hypersurrogatemodel import TrainableLLM, ComparisonTuner
from hypersurrogatemodel.utils import Logger

# Set up logger
logger = Logger("quick_start_comparison")

def quick_start():
    """快速開始：比對數據集並進行調優"""
    # 1. 準備數據集 (Prepare dataset)
    # 您的數據集應該是JSON格式，包含輸入文本和期望答案
    sample_dataset = [
        {
            "text": "選擇適合空氣動力學模擬的代理模型，數據點1000個，5個輸入變量",
            "answer": "神經網路"
        },
        {
            "text": "結構優化問題，500個樣本，3個設計變量，預算有限",
            "answer": "多項式回歸"
        },
        {
            "text": "複雜非線性系統，2000個數據點，8個輸入變量，需要不確定性量化",
            "answer": "高斯過程"
        }
    ]
    
    # 保存數據集
    dataset_path = "my_dataset.json"
    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump(sample_dataset, f, indent=2, ensure_ascii=False)
    logger.success(f"數據集已創建: {dataset_path}")
    
    # 2. 初始化模型 (Initialize model)
    logger.step("初始化模型...")
    model = TrainableLLM(
        base_model_name="google/gemma-3-270m-it",
        use_lora=True  # 使用LoRA進行高效微調
    )
    tokenizer = model.get_tokenizer()
    
    # 3. 創建比對調優器 (Create comparison tuner)
    logger.step("創建比對調優器...")
    tuner = ComparisonTuner(
        model=model,
        tokenizer=tokenizer,
        output_dir="./tuning_results",
        save_files=False  # 禁用檔案保存以減少輸出
    )
    
    # 4. 載入數據集並比對差異 (Load dataset and compare differences)
    logger.info("分析模型表現與數據集答案的差異...")
    comparison_results = tuner.load_and_compare_dataset(
        dataset_path=dataset_path,
        text_column="text",      # 輸入文本的欄位名
        answer_column="answer",   # 正確答案的欄位名
        task_type="generation",   # 任務類型：generation 或 classification
        comparison_method="similarity"  # 比對方法：exact_match, similarity, structured
    )
    
    # 5. 查看比對結果 (View comparison results)
    metrics = comparison_results['overall_metrics']
    logger.result("初始表現:")
    logger.info(f"準確率: {metrics['accuracy']:.3f}")
    logger.info(f"平均相似度: {metrics['average_similarity']:.3f}")
    logger.info(f"正確預測: {metrics['correct_predictions']}")
    logger.info(f"錯誤預測: {metrics['incorrect_predictions']}")
    
    # 顯示錯誤示例
    if comparison_results['differences']:
        logger.warning("錯誤示例:")
        for i, diff in enumerate(comparison_results['differences'][:2]):
            logger.info(f"示例 {i+1}:")
            logger.info(f"輸入: {diff['input_text'][:60]}...")
            logger.info(f"預測: {diff['prediction']}")
            logger.info(f"正確答案: {diff['ground_truth']}")
            logger.info(f"相似度: {diff['similarity']:.3f}")
    
    # 6.adaptive tuning
    if metrics['incorrect_predictions'] > 0:
        logger.step("開始自適應調優...")
        tuning_results = tuner.adaptive_tuning(
            comparison_results=comparison_results,
            dataset_path=dataset_path,
            text_column="text",
            answer_column="answer",
            tuning_strategy="error_focused",  
            max_epochs=2,                    # train epochs
            learning_rate=1e-5,              
        )
        
        # 7. 查看調優結果 (View tuning results)
        logger.success("調優完成!")
        pre_metrics = tuning_results['pre_tuning_metrics']
        post_metrics = tuning_results['post_tuning_metrics']
        
        logger.result(f"調優前準確率: {pre_metrics['accuracy']:.3f}")
        logger.result(f"調優後準確率: {post_metrics['accuracy']:.3f}")
        
        improvement = post_metrics['accuracy'] - pre_metrics['accuracy']
        logger.result(f"準確率提升: {improvement:+.3f}")
        
        error_reduction = tuning_results['improvement_analysis']['error_reduction']
        logger.result(f"錯誤減少: {error_reduction} 個樣本")
        
        # 8. 保存調優後的模型 (Save tuned model)
        model_save_path = "./my_tuned_model"
        model.save_model(model_save_path)
        logger.success(f"調優後模型已保存到: {model_save_path}")
        
    else:
        logger.success("模型在此數據集上已經表現完美!")
    
    logger.info("結果文件保存在: ./tuning_results/")
    logger.success("快速開始完成!")

def test_your_own_dataset():
    """測試您自己的數據集"""
    
    logger.info("=" * 50)
    logger.info("💡 使用您自己的數據集")
    logger.info("=" * 50)
    
    logger.info("""
要使用您自己的數據集，請按照以下格式準備JSON文件：

{
  "data": [
    {
      "text": "您的輸入文本",
      "answer": "期望的答案"
    },
    {
      "text": "另一個輸入文本", 
      "answer": "另一個期望答案"
    }
  ]
}

然後修改以下代碼中的參數：

```python
# 載入您的數據集
comparison_results = tuner.load_and_compare_dataset(
    dataset_path="您的數據集路徑.json",
    text_column="text",      # 您的輸入文本欄位名
    answer_column="answer",   # 您的答案欄位名
    task_type="generation",   # 或 "classification"
    comparison_method="similarity"  # 或 "exact_match", "structured"
)

# 進行調優
tuning_results = tuner.adaptive_tuning(
    comparison_results=comparison_results,
    dataset_path="您的數據集路徑.json",
    text_column="text",
    answer_column="answer",
    tuning_strategy="error_focused",  # 或 "full_retrain", "incremental"
    max_epochs=3,
    learning_rate=2e-5,
)
```

調優策略說明：
- error_focused: 只使用錯誤樣本進行訓練（推薦）
- incremental: 使用錯誤樣本 + 部分正確樣本
- full_retrain: 使用全部數據重新訓練

比對方法說明：
- exact_match: 完全匹配
- similarity: 基於相似度比對（推薦用於文本生成）
- structured: 結構化比對（用於JSON等格式化輸出）
    """)

if __name__ == "__main__":
    quick_start()
