import json 
from typing import Optional, List, Dict, Any
import torch

from hypersurrogatemodel import Logger, TrainableLLM

logger = Logger("Pipelined-runner")
torch.set_float32_matmul_precision('high')

if torch.cuda.is_available():
    torch.cuda.empty_cache()


model = TrainableLLM(load_type="from_pretrained")
model.train()
model.acc_trainer()
