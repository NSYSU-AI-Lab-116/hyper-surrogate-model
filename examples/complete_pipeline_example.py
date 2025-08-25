"""
Complete Training Pipeline Example

This example demonstrates the complete training pipeline using the TrainingManager
for end-to-end model training with domain-specific data and prompt engineering.
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
    Complete training pipeline demonstration.
    """
    # Initialize logger
    logger = Logger("complete_pipeline")
    logger.info("Starting complete training pipeline example")
    
    # Set random seed
    set_random_seed(42)
    
    # Create experiment directory
    experiment_dir = create_experiment_directory(
        base_dir="./experiments",
        experiment_name="complete_pipeline",
        timestamp=True
    )
    
    # Step 1: Initialize Training Manager
    logger.info("Initializing Training Manager...")
    training_manager = TrainingManager(
        base_model_name="google/gemma-3-270m-it",
        output_dir=str(experiment_dir)
    )
    
    # Step 2: Prepare domain-specific dataset with advanced prompt engineering
    logger.info("Preparing domain-specific dataset...")
    
    # Create custom prompt template for your domain
    prompt_template = PromptTemplate("domain_specific")
    prompt_template.add_custom_template(
        name="technical_classification",
        template="""
領域：技術分類分析

任務描述：
請分析以下技術相關文本，識別其所屬的技術領域和關鍵特徵。

輸入文本：
{text}

分析要求：
1. 識別主要技術領域
2. 提取關鍵技術詞彙
3. 判斷技術成熟度
4. 評估應用場景

分類結果："""
    )
    
    # Initialize dataset processor with custom template
    tokenizer_temp = training_manager._get_tokenizer() # type: ignore
    dataset_processor = DomainDatasetProcessor(
        tokenizer=tokenizer_temp,
        max_length=512,
        prompt_template=prompt_template
    )
    
    # Create comprehensive dataset
    dataset = create_comprehensive_dataset(dataset_processor)
    
    # Step 3: Configure training parameters
    model_config = {
        "hidden_size": 512,  # 增加隱藏層大小
        "dropout_rate": 0.3,  # 增加 dropout 以防過擬合
        "use_lora": True,
        "lora_config": {
            "r": 32,  # 增加 LoRA rank
            "lora_alpha": 64,  # 增加 alpha
            "lora_dropout": 0.1,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        }
    }
    
    training_config = {
        "output_dir": str(experiment_dir / "training"),
        "num_train_epochs": 10,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 20,
        "learning_rate": 1e-4,  # 降低學習率到更保守的值
        "weight_decay": 0.01,
        "logging_steps": 10,
        "evaluation_strategy": "epoch",
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
    
    # Step 6: Demonstrate the trained model
    logger.info("Demonstrating trained model...")
    demonstrate_trained_model(
        classification_results["model"],
        tokenizer_temp,
        prompt_template
    )
    
    logger.info("Complete training pipeline finished successfully!")
    
    return {
        "experiment_dir": experiment_dir,
        "classification_results": classification_results,
        "config": config_to_save,
    }


def create_comprehensive_dataset(dataset_processor):
    """
    Create a comprehensive 12-class dataset with domain-specific prompts.
    
    Args:
        dataset_processor: DomainDatasetProcessor instance
        
    Returns:
        Processed dataset
    """
    # Extended dataset with more samples per class
    class_data = {
        # Technology (0)
        0: [
            "人工智慧演算法正在改變軟體開發的方式",
            "雲端運算提供可擴展的基礎設施解決方案",
            "區塊鏈技術確保資料的去中心化安全性",
            "量子電腦將徹底改變密碼學和計算能力",
            "物聯網設備實現了智慧家居自動化控制",
            "5G網路技術提供超高速無線連接能力",
            "邊緣計算減少了資料處理的延遲時間",
            "機器學習模型在圖像識別領域表現卓越",
            "深度學習框架加速了AI應用開發",
            "自動駕駛技術正在重塑交通運輸行業",
            "虛擬實境創造了沉浸式數位體驗",
            "網路安全防護對抗日益增長的威脅",
            "軟體即服務模式改變了企業IT架構",
            "開源社群推動了技術創新和協作",
            "程式碼版本控制提高了開發效率",
            "微服務架構提升了系統的可擴展性",
        ],
        
        # Healthcare (1)
        1: [
            "基因編輯技術為治療遺傳疾病帶來希望",
            "遠程醫療服務改善了偏遠地區的醫療可及性",
            "個人化醫療根據基因資訊制定治療方案",
            "醫療機器人輔助進行精密外科手術",
            "穿戴式設備持續監測患者的生理指標",
            "免疫療法在癌症治療中顯示出突破性進展",
            "數位病理學提高了疾病診斷的準確性",
            "臨床試驗數據驅動新藥研發流程",
            "醫療影像AI協助醫生進行精準診斷",
            "電子病歷系統整合了患者健康資訊",
            "預防醫學強調疾病的早期發現",
            "康復治療幫助患者恢復身體功能",
            "公共衛生政策保護社區健康安全",
            "醫療器械創新改善了治療效果",
            "藥物基因組學指導個人化用藥",
            "心理健康服務提供情緒支持和治療",
            "心理健康應用程式提供便捷的心理支持服務",
        ],
        
        # Finance (2)
        2: [
            "加密貨幣市場波動性影響投資者決策",
            "數位銀行服務革命性地改變金融交易方式",
            "演算法交易在現代金融市場中占主導地位",
            "去中心化金融平台挑戰傳統銀行體系",
            "行動支付技術簡化了日常消費流程",
            "人工智慧風險評估提高了信貸決策效率",
            "保險科技創新改善了客戶服務體驗",
            "央行數位貨幣正在全球範圍內試點推行",
            "股票市場分析依賴大數據和機器學習",
            "財富管理平台提供個人化投資建議",
            "區塊鏈技術確保金融交易的透明度",
            "眾籌平台連接投資者和創業項目",
            "金融科技監管促進創新與風險平衡",
            "智能合約自動執行金融協議條款",
            "信用評分模型預測借款人還款能力",
            "資產管理公司運用量化策略優化投資組合",
        ],
        
        # Education (3)
        3: [
            "線上學習平台擴大了教育資源的可及性",
            "適應性學習系統個人化教育內容推薦",
            "虛擬實境技術創造沉浸式學習體驗",
            "遊戲化教學方法顯著提高學生參與度",
            "人工智慧導師提供24小時學習支援",
            "區塊鏈技術確保學歷認證的真實性",
            "MOOC課程打破了地理和時間的學習限制",
            "數據分析幫助教師了解學生學習進度",
            "翻轉教室模式改變傳統師生互動方式",
            "微學習策略將知識分解為易消化片段",
            "同儕學習網絡促進知識共享交流",
            "教育科技創新提升學習成效評估",
            "多元智能理論指導個人化教學方法",
            "終身學習概念適應快速變化的職場需求",
            "數位素養教育培養21世紀關鍵技能",
            "混合式學習結合線上線下教學優勢",
        ],
        
        # Entertainment (4)
        4: [
            "串流媒體平台競爭激烈爭奪獨家內容版權",
            "虛擬實境遊戲創造前所未有的沉浸體驗",
            "人工智慧生成音樂和藝術作品",
            "社交媒體平台重新定義娛樂內容分發",
            "電競產業快速發展吸引大量觀眾和投資",
            "互動式電影讓觀眾參與劇情發展決策",
            "擴增實境技術增強現實世界娛樂體驗",
            "播客和音頻內容成為新興娛樂媒體",
            "網路直播平台創造新的娛樂互動模式",
            "虛擬偶像利用數位技術打造全新娛樂體驗",
            "沉浸式劇場結合科技與表演藝術",
            "遊戲化健身應用讓運動變得有趣",
            "短影片平台改變內容消費習慣",
            "人工智慧推薦系統個人化娛樂內容",
            "線上音樂會突破地理限制連接全球觀眾",
            "數位藝術NFT創造新的藝術品收藏市場",
        ],
        
        # Sports (5)
        5: [
            "運動分析學優化運動員訓練和比賽策略",
            "穿戴式設備追蹤運動員的生理數據",
            "虛擬實境訓練系統模擬真實比賽場景",
            "電子競技成為正式的體育競賽項目",
            "運動醫學預防和治療運動相關傷害",
            "數據驅動的球隊管理提高競爭優勢",
            "體育博彩技術平台提供即時賠率分析",
            "運動心理學幫助運動員提升心理素質",
            "生物力學分析改善運動員技術動作",
            "運動營養科學優化運動員體能表現",
            "智慧體育場館提升觀眾觀賽體驗",
            "運動復健科技加速傷後恢復過程",
            "青少年體育發展培養未來運動人才",
            "殘疾人運動推廣促進社會包容性",
            "運動直播技術讓全球觀眾共享精彩賽事",
            "運動科學研究提升人體運動極限",
        ],
        
        # Politics (6)
        6: [
            "選舉改革旨在提高選民參與率和透明度",
            "國際外交關係影響全球政治穩定格局",
            "政策制定者討論氣候變遷法律優先事項",
            "數位政府服務提高公共行政效率",
            "社交媒體對政治輿論形成產生重大影響",
            "公民參與平台促進民主決策過程",
            "政治極化現象挑戰傳統政治體系",
            "國際貿易政策影響全球經濟合作關係",
            "立法程序改革提升國會運作效率",
            "政治透明度法案促進政府問責制",
            "跨黨派合作解決重大社會議題",
            "地方政府創新提升公共服務品質",
            "選舉科技確保投票過程安全透明",
            "政治教育培養公民的民主素養",
            "國際制裁政策維護全球秩序",
            "政府數位轉型提升行政服務效能",
        ],
        
        # Science (7)
        7: [
            "研究人員在遙遠星系中發現新的系外行星",
            "基因編輯技術治療遺傳性疾病取得突破",
            "氣候模型預測未來環境變化趨勢",
            "量子物理學研究揭示宇宙基本運作規律",
            "幹細胞療法為組織再生帶來新希望",
            "深海探索發現新的海洋生物物種",
            "材料科學創新開發超導體新材料",
            "天體物理學家觀測到黑洞合併事件",
            "人工智慧協助科學研究數據分析",
            "粒子物理學實驗探索宇宙基本粒子",
            "生物技術創新推動精準醫療發展",
            "奈米科技在材料工程中的突破應用",
            "考古學新發現改寫人類歷史認知",
            "海洋科學研究揭示氣候變遷機制",
            "空間科學任務探索太陽系邊界",
            "認知科學研究大腦意識形成機制",
        ],
        
        # Travel (8)
        8: [
            "永續旅遊實踐保護自然環境和文化遺產",
            "數位遊民在異國他鄉遠程工作生活",
            "文化沉浸體驗豐富旅行者的人生閱歷",
            "智慧旅遊技術提供個人化行程推薦",
            "生態旅遊促進環境保護和社區發展",
            "虛擬旅遊體驗讓人們足不出戶探索世界",
            "太空旅遊即將成為富裕人群的新選擇",
            "旅遊保險科技保障旅行者的安全權益",
            "文化遺產旅遊推廣歷史文化保護",
            "探險旅遊挑戰旅行者的身心極限",
            "美食旅遊體驗不同文化的料理精髓",
            "醫療旅遊結合健康療養與旅行體驗",
            "攝影旅遊記錄世界各地的美麗風景",
            "背包客文化倡導簡約自由的旅行方式",
            "豪華旅遊提供頂級舒適的度假體驗",
            "主題樂園創造歡樂難忘的家庭回憶",
        ],
        
        # Food (9)
        9: [
            "植物性蛋白質提供永續營養替代方案",
            "分子美食學革命性地改變傳統烹飪方法",
            "在地食物運動支持社區農業發展",
            "人工智慧優化食品供應鏈管理效率",
            "細胞培養肉技術減少環境影響",
            "營養基因學個人化飲食健康管理",
            "食品區塊鏈確保食品來源可追溯性",
            "垂直農業技術在城市環境中生產新鮮農產品",
            "功能性食品提供額外的健康益處",
            "傳統發酵技術創造獨特風味食品",
            "有機農業堅持天然無化學添加原則",
            "食品科技創新延長食品保存期限",
            "素食主義推廣健康環保的飲食理念",
            "烘焙藝術結合科學與創意表現",
            "飲食文化交流促進國際友誼發展",
            "食品安全檢測保障消費者健康權益",
        ],
        
        # Fashion (10)
        10: [
            "永續時尚品牌使用環保材料和生產工藝",
            "3D列印技術創造客製化服裝設計",
            "時裝週展示新興設計師的創意才華",
            "快時尚產業面臨環境責任和道德挑戰",
            "虛擬試衣技術改善線上購物體驗",
            "智慧紡織品集成電子功能和傳統面料",
            "循環經濟模式重新定義時尚產業價值鏈",
            "人工智慧預測時尚趨勢和消費者偏好",
            "高級訂製時裝展現精湛手工藝技術",
            "街頭時尚文化影響主流服裝設計",
            "復古風格回歸帶動懷舊時尚潮流",
            "運動時尚結合功能性與美觀設計",
            "時尚攝影藝術記錄服裝美學瞬間",
            "個人形象顧問提供專業穿搭建議",
            "奢侈品牌維護傳統工藝和品牌價值",
            "二手時尚市場推廣可持續消費理念",
        ],
        
        # Environment (11)
        11: [
            "再生能源技術大幅減少碳排放量",
            "海洋保育努力保護海洋生態系統",
            "植樹造林計畫對抗氣候變遷效應",
            "循環經濟減少廢棄物和資源浪費",
            "生物多樣性保護維護地球生態平衡",
            "綠色建築技術提高能源使用效率",
            "環境監測技術追蹤污染和生態變化",
            "碳捕捉和儲存技術減緩全球暖化",
            "環境教育培養公眾生態保護意識",
            "濕地復育計畫恢復自然生態棲息地",
            "綠色交通減少城市空氣污染",
            "野生動物保護維護生物多樣性",
            "環境法律規範企業環保責任",
            "清潔技術創新解決環境污染問題",
            "永續農業實踐保護土壤和水資源",
            "氣候適應策略應對環境變化挑戰",
        ],
    }
    
    # Flatten the data
    texts = []
    labels = []
    
    for label, text_list in class_data.items():
        texts.extend(text_list)
        labels.extend([label] * len(text_list))
    
    # Create dataset using the enhanced prompt template
    raw_data = [{"input_data": text, "target": label} for text, label in zip(texts, labels)]
    
    # Create domain-specific dataset first to get formatted prompts
    domain_dataset = dataset_processor.create_domain_specific_dataset(
        data=raw_data,
        domain="technical_classification",
        task_description="分析技術文本並進行12類分類",
        data_key="input_data",
        target_key="target"
    )
    
    # Extract prompts and labels for classification dataset
    prompts = [item["prompt"] for item in domain_dataset]
    targets = [item["target"] for item in domain_dataset]
    
    # Convert to classification dataset with proper formatting
    classification_dataset = dataset_processor.create_classification_dataset(
        texts=prompts,
        labels=targets,
        domain="technical_classification",
        include_prompt=False  # Already formatted
    )
    
    # Tokenize dataset for model input
    dataset = dataset_processor.tokenize_dataset(
        dataset=classification_dataset,
        text_column="text",
        label_column="label"
    )
    
    return dataset


def demonstrate_trained_model(model, tokenizer, prompt_template):
    """
    Demonstrate the trained model with inference examples.
    
    Args:
        model: Trained model
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
    print("TRAINED MODEL DEMONSTRATION")
    print("="*70)
    
    for text in test_examples:
        # Create domain-specific prompt
        formatted_prompt = prompt_template.format_prompt(
            template_type="technical_classification",
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
        print(f"\nComplete training pipeline finished successfully!")
        print(f"Experiment directory: {results['experiment_dir']}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
