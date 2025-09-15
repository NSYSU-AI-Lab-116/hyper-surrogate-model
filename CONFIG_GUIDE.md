# HyperSurrogateModel 配置系統說明

## 🎯 概覽

使用 YAML 配置文件來控制模型的所有參數，簡單直觀！

## 📋 支援的配置參數

### 模型配置 (model)
- `pretrained_model`: 預訓練模型名稱 (如 "google/gemma-2-2b-it")
- `use_lora`: 是否使用LoRA微調 (true/false)
- `device`: 運算設備 ("cuda"/"cpu"/"auto")

### 生成配置 (generation)
- `max_new_tokens`: 最大生成token數 (預設: 50)
- `temperature`: 溫度參數，控制隨機性 (預設: 0.7)
- `do_sample`: 是否使用採樣 (預設: true)
- `top_k`: Top-k採樣參數 (預設: 64)
- `top_p`: Top-p (nucleus) 採樣參數 (預設: 0.95)
- `repetition_penalty`: 重複懲罰 (預設: 1.1)
- `length_penalty`: 長度懲罰 (預設: 1.0)
- `num_beams`: Beam search數量 (預設: 1)

### LoRA配置 (lora)
- `r`: LoRA rank (預設: 16)
- `lora_alpha`: LoRA alpha (預設: 32)
- `lora_dropout`: LoRA dropout (預設: 0.1)
- `target_modules`: 目標模組列表

### 訓練配置 (training)
- `batch_size`: 批次大小 (預設: 8)
- `learning_rate`: 學習率 (預設: 2e-5)
- `num_epochs`: 訓練輪數 (預設: 3)
- 等等...

## 🔧 使用方法

### 主配置文件: config.yaml

編輯 `config.yaml` 文件來設定您的參數：

```yaml
# 模型設定
model:
  pretrained_model: "microsoft/DialoGPT-medium"
  use_lora: true
  device: "auto"

# 生成參數
generation:
  max_new_tokens: 150
  temperature: 0.9
  top_k: 40
  top_p: 0.8
  do_sample: true

# LoRA設定
lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.1
  target_modules:
    - "q_proj"
    - "v_proj"
    - "k_proj"
    - "o_proj"

# 訓練參數
training:
  batch_size: 4
  learning_rate: 2e-5
  num_epochs: 3
```

### 使用模型

```python
from hypersurrogatemodel import TrainableLLM

# 自動讀取 config.yaml
model = TrainableLLM()

# 或指定自定義配置文件
model = TrainableLLM()  # 會自動使用 config.yaml
```

## 📊 配置方式

**唯一配置來源**: YAML 配置文件 (`config.yaml`)
- 簡單直觀的階層式結構
- 支援中文註解
- 類型安全，不需要字串轉換
- 版本控制友好

## 🧪 測試配置

運行示例程式來測試配置系統：

```bash
cd /home/alvin/hyper-surrogate-model
uv run python example_config_usage.py
```

## 📝 配置文件範例

完整的 `config.yaml` 範例：

```yaml
# Hyper Surrogate Model Configuration
model:
  pretrained_model: "google/gemma-2-2b-it"
  use_lora: true
  device: "auto"

generation:
  max_new_tokens: 50
  temperature: 0.7
  do_sample: true
  top_k: 64
  top_p: 0.95
  repetition_penalty: 1.1
  length_penalty: 1.0
  num_beams: 1

training:
  batch_size: 8
  learning_rate: 2e-5
  num_epochs: 3
  warmup_steps: 100
  weight_decay: 0.01
  fp16: false

lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.1
  target_modules:
    - "q_proj"
    - "v_proj"
    - "k_proj"
    - "o_proj"

logging:
  level: "INFO"
  use_wandb: false
  wandb_project: "hypersurrogatemodel"
  save_files: true
  output_dir: "./results"

dataset:
  max_length: 512
  padding: "max_length"
  truncation: true
  template_type: "generation"

comparison:
  method: "similarity"
  batch_size: 256
  similarity_threshold: 0.8
  tuning_strategy: "error_focused"
```

## 🎉 完成！

現在您只需要編輯 `config.yaml` 就能控制所有模型參數，簡單明瞭！
