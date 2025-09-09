from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
import json 
from hypersurrogatemodel import Logger

logger = Logger("QUickInterface")
logger.setFunctionsName("eval")

local_model_path = "./saved_model"
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(local_model_path)

class ModelWithCustomHead(nn.Module):
    def __init__(self, base_model, custom_head_path):
        super().__init__()
        self.base_model = base_model
        
        # head layer (load weights and structure from file)
        custom_head_state = torch.load(custom_head_path)
        hidden_size = base_model.config.hidden_size
        self.numerical_head = nn.Sequential(
            nn.Linear(hidden_size, custom_head_state['0.weight'].shape[0]),  # Layer 0
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(custom_head_state['0.weight'].shape[0], 1)  # Layer 3
        )
        self.numerical_head.load_state_dict(custom_head_state)
    
    def forward(self, input_ids, **kwargs):
        # Get hidden states from base model
        outputs = self.base_model(input_ids, output_hidden_states=True, **kwargs)
        
        # Use last hidden state for your custom head
        last_hidden_state = outputs.hidden_states[-1]
        
        # Apply your numerical head
        numerical_output = self.numerical_head(last_hidden_state[:, -1, :]) 
        
        return {
            'logits': outputs.logits,
            'numerical_output': numerical_output,
            'hidden_states': outputs.hidden_states
        }

combined_model = ModelWithCustomHead(model, f"{local_model_path}/numerical_head.pt")
combined_model.eval()

with open("./data/processed/NAS_bench_201/cifar10_cleaned.json", "r") as f:
    data = json.load(f)

results = []
for item in data[:1]:
    print(f"Processing item: {item['text']}...")
    input_ids = tokenizer(item['text'], return_tensors="pt").input_ids
    
    with torch.no_grad():
        prediction = combined_model(input_ids)
        result = prediction['numerical_output'].cpu().numpy().tolist()
        
        results.append({
            'input': item['text'],
            'prediction': result
        })
        
        logger.info(f"Input: {item['text'][:50]}...")
        logger.info(f"Prediction: {result}")

with open("./data/results/predictions.json", "w") as f:
    json.dump(results, f, indent=2)