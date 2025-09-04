# Dataset Comparison and Adaptive Tuning

本功能實現了LLM載入數據集、生成預測、與正確答案比對差異，並進行自適應調優的完整流程。

## 功能特色

✅ **數據集載入與處理**: 支援JSON、CSV、JSONL格式  
✅ **智能預測生成**: 支援分類和文本生成任務  
✅ **多種比對方法**: 精確匹配、相似度比對、結構化比對  
✅ **自適應調優策略**: 錯誤聚焦、增量學習、完全重訓  
✅ **詳細分析報告**: 錯誤分析、改進追蹤、性能監控  
✅ **LoRA高效微調**: 參數效率高、記憶體需求低  

## 🎯 完整工作流程

```txt
1. 數據準備 → 2. 載入比對 → 3. 分析差異 → 4. 自適應調優 → 5. 驗證改進
     ↓              ↓              ↓              ↓              ↓
  JSON格式      LLM預測vs答案    錯誤類型分析    LoRA微調      性能提升
```

## 快速開始

### 1. 基本使用

```python
from hypersurrogatemodel import TrainableLLM, ComparisonTuner

# 初始化模型和調優器
model = TrainableLLM(use_lora=True)
tuner = ComparisonTuner(model=model, tokenizer=model.get_tokenizer())

# 載入數據集並比對差異
results = tuner.load_and_compare_dataset(
    dataset_path="your_dataset.json",
    text_column="text",
    answer_column="answer",
    task_type="generation",
    comparison_method="similarity"
)

# 基於差異進行自適應調優
tuning_results = tuner.adaptive_tuning(
    comparison_results=results,
    dataset_path="your_dataset.json",
    tuning_strategy="error_focused",
    max_epochs=3
)
```

### 2. 數據集格式

您的數據集應該是JSON格式：

```json
[
  {
    "text": "輸入文本：選擇適合的代理模型",
    "answer": "神經網路"
  },
  {
    "text": "另一個輸入文本",
    "answer": "期望的答案"
  }
]
```

## 詳細功能說明

### 比對方法 (Comparison Methods)

1. **exact_match**: 完全匹配
   - 適用於分類任務
   - 預測值必須與正確答案完全相同

2. **similarity**: 相似度比對  
   - 適用於文本生成任務
   - 基於token重疊計算相似度
   - 推薦閾值：0.8

3. **structured**: 結構化比對
   - 適用於JSON等格式化輸出
   - 支援部分匹配評分

### 調優策略 (Tuning Strategies)

1. **error_focused**: 錯誤聚焦（推薦）
   - 僅使用錯誤預測的樣本進行訓練
   - 高效且針對性強
   - 適合錯誤率較低的情況

2. **incremental**: 增量學習
   - 使用錯誤樣本 + 部分正確樣本
   - 平衡改進與穩定性
   - 適合中等錯誤率的情況

3. **full_retrain**: 完全重訓
   - 使用全部數據重新訓練
   - 最全面但耗時較長
   - 適合錯誤率很高的情況

### 任務類型 (Task Types)

1. **classification**: 分類任務
   - 輸出固定類別
   - 使用分類損失函數
   - 評估指標：準確率、F1分數

2. **generation**: 文本生成任務
   - 自由文本輸出
   - 使用語言模型損失函數  
   - 評估指標：相似度、困惑度

## 完整示例

### 代理模型選擇示例

```python
import json
from hypersurrogatemodel import TrainableLLM, ComparisonTuner

# 創建代理模型選擇數據集
dataset = [
    {
        "text": "設計空氣動力學模擬的代理模型，1000個數據點，5個輸入變量，需要高精度",
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
with open("surrogate_dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

# 初始化模型
model = TrainableLLM(
    base_model_name="google/gemma-3-270m-it",
    use_lora=True,
    lora_config={
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "v_proj"]
    }
)

# 創建調優器
tuner = ComparisonTuner(
    model=model,
    tokenizer=model.get_tokenizer(),
    output_dir="./results"
)

# 載入並比對
print("分析初始性能...")
results = tuner.load_and_compare_dataset(
    dataset_path="surrogate_dataset.json",
    text_column="text",
    answer_column="answer", 
    task_type="generation",
    comparison_method="similarity"
)

print(f"準確率: {results['overall_metrics']['accuracy']:.3f}")
print(f"錯誤數量: {results['overall_metrics']['incorrect_predictions']}")

# 自適應調優
if results['overall_metrics']['incorrect_predictions'] > 0:
    print("\\n開始自適應調優...")
    tuning_results = tuner.adaptive_tuning(
        comparison_results=results,
        dataset_path="surrogate_dataset.json",
        text_column="text",
        answer_column="answer",
        tuning_strategy="error_focused",
        max_epochs=3,
        learning_rate=1e-5
    )
    
    # 顯示改進
    improvement = tuning_results['improvement_analysis']
    print(f"準確率改進: {improvement['improvements']['accuracy']['absolute_improvement']:+.3f}")
    print(f"錯誤減少: {improvement['error_reduction']} 個樣本")

# 保存調優後的模型
model.save_model("./tuned_surrogate_model")
print("調優完成！模型已保存。")
```

## 輸出結果

### 比對結果示例

```json
{
  "overall_metrics": {
    "accuracy": 0.750,
    "average_similarity": 0.823,
    "total_samples": 8,
    "correct_predictions": 6,
    "incorrect_predictions": 2
  },
  "error_analysis": {
    "error_types": {
      "partial_match": 1,
      "completely_different": 1
    },
    "most_common_error": "partial_match"
  },
  "differences": [
    {
      "index": 2,
      "input_text": "複雜非線性系統...",
      "prediction": "支援向量機",
      "ground_truth": "高斯過程", 
      "similarity": 0.456,
      "difference_type": "completely_different"
    }
  ]
}
```

### 調優結果示例

```json
{
  "tuning_strategy": "error_focused",
  "pre_tuning_metrics": {
    "accuracy": 0.750,
    "incorrect_predictions": 2
  },
  "post_tuning_metrics": {
    "accuracy": 0.875,
    "incorrect_predictions": 1
  },
  "improvement_analysis": {
    "improvements": {
      "accuracy": {
        "absolute_improvement": 0.125,
        "percentage_improvement": 16.7
      }
    },
    "error_reduction": 1
  }
}
```

## 進階配置

### 自定義LoRA配置

```python
model = TrainableLLM(
    use_lora=True,
    lora_config={
        "r": 16,                    # LoRA rank
        "lora_alpha": 32,           # LoRA scaling parameter  
        "lora_dropout": 0.1,        # Dropout rate
        "target_modules": [         # Target modules for LoRA
            "q_proj", "v_proj", 
            "k_proj", "o_proj"
        ],
    }
)
```

### 自定義訓練參數

```python
tuning_results = tuner.adaptive_tuning(
    comparison_results=results,
    dataset_path="dataset.json",
    tuning_strategy="incremental",
    max_epochs=5,
    learning_rate=2e-5,
)
```

### 使用Weights & Biases追蹤

```python
tuner = ComparisonTuner(
    model=model,
    tokenizer=tokenizer,
    output_dir="./results",
    use_wandb=True  # 啟用W&B追蹤
)
```

## 常見問題

### Q: 如何處理大型數據集？

A: 系統自動按批次處理，可以調整 `batch_size` 參數來控制記憶體使用。

### Q: 調優需要多長時間？

A: 取決於數據集大小和模型複雜度。error_focused策略通常最快，因為只訓練錯誤樣本。

### Q: 如何選擇比對方法？

A:

- 分類任務使用 `exact_match`
- 文本生成使用 `similarity`
- JSON輸出使用 `structured`

### Q: 為什麼初始準確率是0.0%？這正常嗎？

A:

**完全正常！** 這實際上證明了調優的必要性和價值：

1. **通用LLM的局限性**：
   - 在通用文本上訓練，缺乏專業工程知識
   - 不了解代理模型選擇的專業術語和邏輯
   - 輸出格式可能不符合預期

2. **專業任務的挑戰**：
   - 需要深度的工程和數值方法知識
   - 要理解不同算法的適用場景
   - 需要給出精確的技術術語

3. **這正是調優的價值**：
   - 將通用智能轉化為專業智能
   - 從"知道很多"到"專精一域"
   - 0% → 60-90% 的改進空間巨大

**期望改進路徑**：初始0% → 第1輪40% → 第2輪70% → 第3輪85%

## 文件結構

調優完成後會生成以下文件：

```text
./comparison_tuning_results/
├── comparison_results_20240101_120000.json    # 比對結果
├── tuning_results_20240101_120000.json        # 調優結果  
├── adaptive_tuning/                           # 訓練輸出
│   ├── checkpoint-100/                        # 模型檢查點
│   └── logs/                                  # 訓練日誌
└── performance_logs/                          # 性能追蹤
    └── performance_log_20240101.json
```

## 相關文件

- `examples/comparison_tuning_example.py` - 完整示例
- `examples/quick_start_comparison.py` - 快速開始指南
- `hypersurrogatemodel/comparison_tuner.py` - 核心實現

---

有任何問題歡迎查看示例代碼或提出issue！
