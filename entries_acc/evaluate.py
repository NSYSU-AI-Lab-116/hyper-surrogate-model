from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
import json
import tqdm
from hypersurrogatemodel import Logger
from hypersurrogatemodel.config import config

logger = Logger("QUickInterface")
logger.setFunctionsName("eval")

local_model_path = config.model.transfer_model_path
if not local_model_path:
    raise ValueError("Please set 'transfer_model_path' in the config file.")
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
            nn.Linear(hidden_size, custom_head_state["0.weight"].shape[0]),  # Layer 0
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(custom_head_state["0.weight"].shape[0], 1),  # Layer 3
        )
        self.numerical_head.load_state_dict(custom_head_state)

    def forward(self, input_ids, **kwargs):
        outputs = self.base_model(input_ids, output_hidden_states=True, **kwargs)

        last_hidden_state = outputs.hidden_states[-1]

        numerical_output = self.numerical_head(last_hidden_state[:, -1, :])

        return {
            "logits": outputs.logits,
            "numerical_output": numerical_output,
            "hidden_states": outputs.hidden_states,
        }


if __name__ == "__main__":
    logger.info("Evaluating model")
    combined_model = ModelWithCustomHead(model, f"{local_model_path}/numerical_head.pt")
    combined_model.eval()
    combined_model.to("cuda" if torch.cuda.is_available() else "cpu")
    datapath = config.dataset.test_data_path
    if not datapath:
        raise ValueError("Please set 'dataset_path' in the config file.")
    with open(datapath, "r") as f:
        data = json.load(f)

    results = []
    logger.info(f"Evaluating {len(data)} items...")

    progress_bar = tqdm.tqdm(data, desc=f"{'Overall':8}")
    for item in data:
        input_ids = tokenizer(item["text"], return_tensors="pt").input_ids
        input_ids = input_ids.to("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            prediction = combined_model(input_ids)
            result = prediction["numerical_output"].cpu().numpy().tolist()

            results.append({"input": item["text"], "prediction": result})
        progress_bar.update(1)
    progress_bar.close()

    logger.info("Evaluation completed! Saving results...")
    with open("./data/results/predictions.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to ./data/results/predictions.json")
