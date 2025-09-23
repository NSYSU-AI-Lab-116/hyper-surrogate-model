from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
import json 
import tqdm
from hypersurrogatemodel import Logger
from hypersurrogatemodel.config import config 
import re
from pathlib import Path
logger = Logger("QUickInterface")
logger.setFunctionsName("eval")

def find_latest_model_version(base_path: str) -> str | None:
    """Finds the latest version directory in the saved model path."""
    p = Path(base_path)
    if not p.exists(): return None
    version_dirs = [d for d in p.iterdir() if d.is_dir() and re.match(r'v\d+', d.name)]
    if not version_dirs: return None
    version_dirs.sort(key=lambda d: int(re.search(r'v(\d+)', d.name).group(1)), reverse=True)
    return str(version_dirs[0])

base_model_path = os.path.abspath("./saved_model")
latest_model_path = find_latest_model_version(base_model_path)
if not latest_model_path:
    logger.error(f"No trained model found in {base_model_path}")
    exit()

logger.info(f"Loading latest model from: {latest_model_path}")
tokenizer = AutoTokenizer.from_pretrained(latest_model_path)
model = AutoModelForCausalLM.from_pretrained(latest_model_path)

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

combined_model = ModelWithCustomHead(model, f"{latest_model_path}/numerical_head.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
combined_model.to(device)
combined_model.eval()

with open("./data/processed/NAS_bench_201/cifar10_test_set.json", "r") as f:
    data = json.load(f)
logger.info(f"read {len(data)} data, start predicting.")


results = []
for item in tqdm(data, desc="Generatingg scores"):
    #print(f"Processing item: {item['text']}...")
    input_ids = tokenizer(item['text'], return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    
    with torch.no_grad():
        prediction_tensor = combined_model(input_ids)
        predicted_score = prediction_tensor.item()
        results.append({
            "text": item['text'],
            "true_answer": float(item['answer']),
            "predicted_score": predicted_score  
        })

output_path = Path(latest_model_path) / "predictions.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

logger.success(f"Evaluation scores saved to {output_path}")