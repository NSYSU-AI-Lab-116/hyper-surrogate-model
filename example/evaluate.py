from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
import json 
from hypersurrogatemodel import Logger
from tqdm import tqdm
import os

logger = Logger("QUickInterface")
logger.setFunctionsName("eval")

local_model_path = os.path.abspath("./saved_model/v5")
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(local_model_path)

norm_params_path = os.path.join(os.path.dirname(local_model_path), "normalization_params.json")
try:
    with open(norm_params_path, 'r') as f:
        norm_params = json.load(f)
    answer_mean = norm_params['mean']
    answer_std = norm_params['std']
    logger.info(f"Successfully loaded normalization parameters from {norm_params_path}")
except FileNotFoundError:
    logger.error(f"Normalization parameters not found at {norm_params_path}. Please run training first.")
    exit()

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
        outputs = self.base_model(input_ids, output_hidden_states=True, **kwargs)
        
        last_hidden_state = outputs.hidden_states[-1]
        
        numerical_output = self.numerical_head(last_hidden_state[:, -1, :]) 
        
        return numerical_output

combined_model = ModelWithCustomHead(model, f"{local_model_path}/numerical_head.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
combined_model.to(device)
combined_model.eval()

with open("./data/processed/NAS_bench_201/cifar10_test_set.json", "r") as f:
    data = json.load(f)
logger.info(f"read {len(data)} data, start predicting.")


results = []
for item in tqdm(data, desc="Predicting"):
    #print(f"Processing item: {item['text']}...")
    input_ids = tokenizer(item['text'], return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    
    with torch.no_grad():
        prediction_tensor = combined_model(input_ids)
        normalized_prediction = prediction_tensor.item() 
        prediction_value = (normalized_prediction * answer_std) + answer_mean
        
        results.append({
            'text': item['text'],
            'answer': item['answer'], 
            'prediction': prediction_value
        })
        

logger.info("Prediction finished.")
with open("./data/results/predictions.json", "w") as f:
    json.dump(results, f, indent=2)