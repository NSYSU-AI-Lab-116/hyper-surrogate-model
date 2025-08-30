"""
Classification Pipeline Example

This example demonstrates a complete classification training pipeline using the TrainingManager
for training a 12-class text classification model with domain-specific prompts.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hypersurrogatemodel import (
    TrainingManager,
    DomainDatasetProcessor,
    PromptTemplate,
    set_random_seed,
    create_experiment_directory,
    Logger,
    DEFAULT_CONFIG
)
from transformers import TrainingArguments
import json


def main():
    """
    Complete classification training pipeline demonstration.
    """
    # Initialize logger
    logger = Logger("classification_pipeline")
    logger.info("Starting classification training pipeline example")
    
    # Set random seed
    set_random_seed(42)
    
    # Create experiment directory
    experiment_dir = create_experiment_directory(
        base_dir="./experiments",
        experiment_name="classification_pipeline",
        timestamp=True
    )
    
    # Step 1: Initialize Training Manager
    logger.info("Initializing Training Manager...")
    training_manager = TrainingManager(
        base_model_name="google/gemma-3-270m-it",
        output_dir=str(experiment_dir)
    )
    
    # Step 2: Prepare classification dataset with domain-specific prompts
    logger.info("Preparing classification dataset...")
    
    # Create custom prompt template for classification
    prompt_template = PromptTemplate("text_classification")
    prompt_template.add_custom_template(
        name="multi_class_classification",
        template="""
        task: classification

        Please classify the following text into one of the predefined categories:

        text:{text}

        classification categories:
        """
    )
    
    # Initialize dataset processor with custom template
    tokenizer_temp = training_manager._get_tokenizer() # type: ignore
    dataset_processor = DomainDatasetProcessor(
        tokenizer=tokenizer_temp,
        max_length=512,
        prompt_template=prompt_template
    )
    
    # Create classification dataset
    dataset = create_classification_dataset(dataset_processor)
    
    # Step 3: Configure model and training parameters
    model_config = {
        "hidden_size": 256,
        "dropout_rate": 0.2,
        "use_lora": True,
        "lora_config": {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        }
    }
    
    training_config = {
        "output_dir": str(experiment_dir / "training"),
        "num_train_epochs": 5,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 2,
        "warmup_steps": 100,
        "learning_rate": 2e-4,
        "weight_decay": 0.01,
        "logging_steps": 10,
        "eval_strategy": "epoch", 
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "accuracy",
        "greater_is_better": True,
        "report_to": None,
        "remove_unused_columns": False,
        "dataloader_pin_memory": False,
    }
    
    # Step 4: Train the classification model
    logger.info("Starting classification model training...")
    classification_results = training_manager.train_classification_model(
        dataset=dataset,
        num_classes=12,
        model_config=model_config,
        training_config=training_config,
    )
    
    # Step 5: Save training configuration and results
    config_to_save = {
        "model_config": model_config,
        "training_config": training_config,
        "dataset_info": {
            "total_samples": len(dataset),
            "num_classes": 12,
            "max_length": 512,
        },
        "results_summary": {
            "final_train_loss": classification_results["results"]["train_results"].training_loss,
            "final_eval_metrics": classification_results["results"]["eval_results"],
            "test_metrics": classification_results["results"].get("test_results", {}),
        }
    }
    
    with open(experiment_dir / "training_config.json", 'w') as f:
        json.dump(config_to_save, f, indent=2, default=str)
    
    # Step 6: Demonstrate the trained classification model
    logger.info("Demonstrating trained classification model...")
    demonstrate_classification_model(
        classification_results["model"],
        tokenizer_temp,
        prompt_template
    )
    
    logger.info("Classification training pipeline finished successfully!")
    
    return {
        "experiment_dir": experiment_dir,
        "classification_results": classification_results,
        "config": config_to_save,
    }


def create_classification_dataset(dataset_processor):
    """
    Create a 12-class text classification dataset.
    
    Args:
        dataset_processor: DomainDatasetProcessor instance
        
    Returns:
        Processed dataset for classification
    """
    # 12-class dataset with balanced samples
    class_data = {
        # Technology (0)
        0: [
            "人工智慧演算法正在改變軟體開發的方式",
            "雲端運算提供可擴展的基礎設施解決方案",
            "區塊鏈技術確保資料的去中心化安全性",
            "量子電腦將徹底改變密碼學和計算能力",
            "物聯網設備實現了智慧家居自動化控制",
        ],
        
        # Healthcare (1)
        1: [
            "基因編輯技術為治療遺傳疾病帶來希望",
            "遠程醫療服務改善了偏遠地區的醫療可及性",
            "個人化醫療根據基因資訊制定治療方案",
            "醫療機器人輔助進行精密外科手術",
            "穿戴式設備持續監測患者的生理指標",
        ],
        
        # Finance (2)
        2: [
            "加密貨幣市場波動性影響投資者決策",
            "數位銀行服務革命性地改變金融交易方式",
            "演算法交易在現代金融市場中占主導地位",
            "去中心化金融平台挑戰傳統銀行體系",
            "行動支付技術簡化了日常消費流程",
        ],
        
        # Education (3)
        3: [
            "線上學習平台擴大了教育資源的可及性",
            "適應性學習系統個人化教育內容推薦",
            "虛擬實境技術創造沉浸式學習體驗",
            "遊戲化教學方法顯著提高學生參與度",
            "人工智慧導師提供24小時學習支援",
        ],
        
        # Entertainment (4)
        4: [
            "串流媒體平台改變了內容消費模式",
            "虛擬演唱會為藝術家提供新的表演舞台",
            "遊戲直播創造了新的娛樂互動體驗",
            "AI生成音樂拓展了創作的可能性",
            "沉浸式電影技術提供前所未有的觀影體驗",
        ],
        
        # Sports (5)
        5: [
            "運動分析數據改善了球員表現評估",
            "虛擬體育賽事吸引了新一代觀眾",
            "穿戴式運動設備監測運動員的生理狀態",
            "電子競技成為全球性的體育項目",
            "運動直播技術讓觀眾體驗更加豐富",
        ],
        
        # Politics (6)
        6: [
            "數位民主平台增進公民政治參與",
            "社群媒體深刻影響政治輿論形成",
            "區塊鏈投票系統提高選舉透明度",
            "大數據分析預測選舉結果趨勢",
            "政治溝通策略適應數位時代特點",
        ],
        
        # Science (7)
        7: [
            "CRISPR基因編輯技術革命性突破",
            "太空探索任務發現新的宇宙秘密",
            "氣候變遷研究揭示地球環境挑戰",
            "量子物理實驗驗證理論預測",
            "生物多樣性研究保護瀕危物種",
        ],
        
        # Travel (8)
        8: [
            "智慧旅遊app個人化行程規劃服務",
            "永續旅遊概念促進環保意識",
            "虛擬旅遊體驗讓人足不出戶遊世界",
            "共享經濟改變住宿和交通方式",
            "文化旅遊深度體驗當地傳統",
        ],
        
        # Food (9)
        9: [
            "植物肉技術提供永續蛋白質來源",
            "精準農業提高食物生產效率",
            "美食delivery app革命性改變餐飲業",
            "分子料理結合科學與藝術創新",
            "食品安全追蹤系統保障消費者健康",
        ],
        
        # Fashion (10)
        10: [
            "永續時尚品牌推動環保理念",
            "3D列印技術製造個人化服裝",
            "虛擬試衣技術改善線上購物體驗",
            "快時尚產業面臨環境責任挑戰",
            "智慧織物整合科技與時尚設計",
        ],
        
        # Environment (11)
        11: [
            "再生能源技術大幅減少碳排放量",
            "海洋保育努力保護海洋生態系統",
            "植樹造林計畫對抗氣候變遷效應",
            "循環經濟減少廢棄物和資源浪費",
            "生物多樣性保護維護地球生態平衡",
        ],
    }
    
    # Flatten the data
    texts = []
    labels = []
    
    for label, text_list in class_data.items():
        texts.extend(text_list)
        labels.extend([label] * len(text_list))
    
    # Create classification dataset directly using the available method
    classification_dataset = dataset_processor.create_classification_dataset(
        texts=texts,
        labels=labels,
        domain="multi_class_classification",
        include_prompt=True  # Use prompts
    )
    
    # Tokenize dataset for model input
    dataset = dataset_processor.tokenize_dataset(
        dataset=classification_dataset,
        text_column="text",
        label_column="label"
    )
    
    return dataset


def demonstrate_classification_model(model, tokenizer, prompt_template):
    """
    Demonstrate the trained classification model with inference examples.
    
    Args:
        model: Trained classification model
        tokenizer: Tokenizer
        prompt_template: Prompt template
    """
    import torch
    
    model.eval()
    
    class_names = [
        "Technology", "Healthcare", "Finance", "Education",
        "Entertainment", "Sports", "Politics", "Science", 
        "Travel", "Food", "Fashion", "Environment"
    ]
    
    test_examples = [
        "ChatGPT等大型語言模型正在改變人機互動方式",
        "新冠疫苗的研發展現了現代醫學的突破性進展",
        "比特幣價格波動反映了數位資產市場的不確定性",
        "線上教育平台讓學習變得更加靈活和便利",
        "Netflix原創內容策略重新定義了娛樂產業",
    ]
    
    print("\n" + "="*70)
    print("CLASSIFICATION MODEL DEMONSTRATION")
    print("="*70)
    
    for text in test_examples:
        # Create classification prompt
        formatted_prompt = prompt_template.format_prompt(
            template_type="multi_class_classification",
            text=text
        )
        
        # Tokenize
        inputs = tokenizer(
            formatted_prompt,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        
        # Get prediction
        with torch.no_grad():
            # Move inputs to the same device as model
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}
            
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            
            logits = outputs["logits"]
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(logits, dim=-1).item()
            confidence = probabilities[0][predicted_class].item() # type: ignore
        
        print(f"\n輸入文本: {text}")
        print(f"預測類別: {class_names[predicted_class]}") # type: ignore
        print(f"信心度: {confidence:.3f}")
        
        # Show top 3 predictions
        top_probs, top_indices = torch.topk(probabilities[0], 3)
        print("前三名預測:")
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            print(f"  {i+1}. {class_names[idx]}: {prob:.3f}")
        print("-" * 50)


def _get_tokenizer(self):
    """Helper method to get tokenizer."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(self.base_model_name)

# Monkey patch the method
TrainingManager._get_tokenizer = _get_tokenizer # type: ignore


if __name__ == "__main__":
    try:
        results = main()
        print(f"\nClassification training pipeline finished successfully!")
        print(f"Experiment directory: {results['experiment_dir']}")
        print(f"Model trained on 12 classes")
        # Print training results summary
        if 'results_summary' in results['config']:
            summary = results['config']['results_summary']
            print(f"Final training loss: {summary.get('final_train_loss', 'N/A')}")
            if 'final_eval_metrics' in summary:
                eval_metrics = summary['final_eval_metrics']
                print(f"Final accuracy: {eval_metrics.get('eval_accuracy', 'N/A'):.4f}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
