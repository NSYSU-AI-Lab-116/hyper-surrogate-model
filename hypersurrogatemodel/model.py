import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from typing import Any, Literal
import json

import torch
import torch.nn as nn
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from accelerate import Accelerator
from torch.utils.data import Dataset
from .utils import Logger , get_gpu_utilization as gpu_util
from .config import config
import huggingface_hub
import dotenv 

logger = Logger("model")
torch.set_float32_matmul_precision("high")

dotenv.load_dotenv()
if os.getenv("huggingface_hub_api_token") is not None:
    huggingface_hub.login(token=os.getenv("huggingface_hub_api_token"))

class TrainingDataset(Dataset):
    """Custom dataset for accelerate training"""

    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_template = "uid:{uid}, structure:{unified_text_description}"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = self.input_template.format(
            uid=sample["uid"],
            unified_text_description=sample["unified_text_description"],
        )
        target = float(sample["true_acc"])

        # Tokenize on-the-fly (accelerate will handle batching)
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "numerical_target": torch.tensor(target, dtype=torch.float32),
        }


def collate_fn(batch):
    """Custom collate function for batch processing"""
    # Extract components
    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    targets = torch.stack([item["numerical_target"] for item in batch]).unsqueeze(1)

    # Pad sequences to max length in batch
    max_len = max(len(ids) for ids in input_ids)

    # Pad input_ids and attention_masks
    padded_input_ids = []
    padded_attention_masks = []

    for ids, mask in zip(input_ids, attention_masks):
        pad_len = max_len - len(ids)
        padded_ids = torch.cat([ids, torch.zeros(pad_len, dtype=torch.long)])
        padded_mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)])

        padded_input_ids.append(padded_ids)
        padded_attention_masks.append(padded_mask)

    return {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_masks),
        "numerical_targets": targets,
    }


class TrainableLLM(nn.Module):
    """
    Trainable text generation model
    """

    def __init__(
        self,
        load_type: Literal["from_pretrained", "from_saved"],
        use_lora: bool | None = None,
        num_outputs: int = 1,
        loss_fn: Any | None = None,
    ):
        """
        Initialize the trainable language model.

        Args:
            base_model_name: Name of the base model to load (uses config if None)
            use_lora: Whether to use LoRA for parameter-efficient fine-tuning (uses config if None)
            lora_config: Custom LoRA configuration dictionary (uses config if None)
            num_outputs: Number of numerical outputs (for regression tasks)
        """
        super().__init__()
        self.use_lora = use_lora if use_lora is not None else False
        self.num_outputs = num_outputs
        self.load_type = load_type
        self.load_model(load_type=load_type)
        self.loss_fn = loss_fn if loss_fn is not None else self.loss_fn
        logger.info("Model initialization complete.")

    def forward(self, input_ids, attention_mask=None, numerical_targets=None, **kwargs):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )
        
        # Numerical prediction
        if not hasattr(outputs, "hidden_states"):
            raise RuntimeError("Addition hidden layer not working")
        hidden_states = outputs.hidden_states[-1]

        if hidden_states is None:
            return result

        if attention_mask is None:
            last_hidden_states = hidden_states[:, -1, :]
            logger.warning("Attention mask not provided; using last token for pooling.")
        else:
            attn = attention_mask.to(hidden_states.device)
            seq_lens = attn.long().sum(dim=1).clamp(min=1) - 1  # shape (B,)
            batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
            last_hidden_states = hidden_states[batch_idx, seq_lens, :]  # shape (B, H)
            
        last_hidden_states = last_hidden_states.to(self.numerical_head[0].weight.device)  # type: ignore # ensure to set self.numerical_head device
        numerical_outputs = self.numerical_head(last_hidden_states)  # type: ignore
        result.update({"numerical_outputs": numerical_outputs})

        # Calculate numerical loss if targets provided
        if numerical_targets is not None:
            numerical_targets = numerical_targets.to(numerical_outputs.device).float()
            numerical_loss = self.loss_fn(numerical_outputs, numerical_targets)
            result["loss"] = numerical_loss
        else:
            logger.error("Numerical loss is None despite targets being provided.")
            
        if torch.isnan(result["loss"]).any() or torch.isinf(result["loss"]).any():
            logger.error("Loss is NaN or Inf")
            result.update({"loss": 0.0})
        
        return result

    def loss_fn(self, predictions, targets):
        if predictions.shape != targets.shape:
            raise ValueError(
                f"Predictions shape {predictions.shape} does not match targets shape {targets.shape}"
            )
        return nn.MSELoss()(predictions, targets)

    def _setup_lora(self, lora_config: dict[str, Any] | None = None) -> None:
        """Set up LoRA configuration"""
        if lora_config is None:
            lora_config = {
                "task_type": TaskType.CAUSAL_LM,
                "r": config.lora.r,  # type: ignore
                "lora_alpha": config.lora.lora_alpha,  # type: ignore
                "lora_dropout": config.lora.lora_dropout,  # type: ignore
                "target_modules": config.lora.target_modules,  # type: ignore
            }
        self.lora_config = lora_config
        lora_config_obj = LoraConfig(**lora_config)
        self.model = get_peft_model(self.model, lora_config_obj)  # type: ignore

    def load_model(self, load_type: Literal["from_pretrained", "from_saved"]) -> None:
        """load model from pretrained or saved path

        Args:
            mode (str): ["from_pretrained", "from_saved"]
        """
        if load_type == "from_pretrained" and config.model.pretrained_model is not None:
            self.model_path = config.model.pretrained_model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,  # type: ignore
                torch_dtype=torch.float32,
                device_map=None,
                trust_remote_code=True,
                attn_implementation="eager",
            )
            logger.info(f"loading model from pretrained: {config.model.pretrained_model}")
            self.hidden_size = self.model.config.hidden_size
            mid = max(1, self.hidden_size // 2)
            self.numerical_head = nn.Sequential(
                nn.Linear(self.hidden_size, mid),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(mid, self.num_outputs),
            )
            logger.info(f"Added numerical output head with {self.num_outputs} outputs")

        elif load_type == "from_saved" and config.model.transfer_model_path is not None:
            self.model_path = config.model.transfer_model_path
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,  # type: ignore
                torch_dtype=torch.float32,
                device_map=None,
                trust_remote_code=True,
                attn_implementation="eager",
            )
            logger.info(f"loading model from saved: {self.model_path}")
            custom_head_state = torch.load(
                f"{self.model_path}/numerical_head.pt", map_location="cpu"
            )

            hidden_mid = custom_head_state["0.weight"].shape[0]
            hidden_size = int(self.model.config.hidden_size)

            self.numerical_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_mid),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_mid, 1),
            )
            self.numerical_head.load_state_dict(custom_head_state)
            logger.info(f"Loaded numerical output head with {self.num_outputs} outputs")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        logger.info(f"Tokenizer loaded: {self.tokenizer.__class__.__name__}")

        logger.info(f"Using LoRA: {self.use_lora}")
        if self.use_lora:
            self._setup_lora()

    def save_model(
        self,
        model_name: str | None = None,
        save_training_state: bool = True,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        epoch: int | None = None,
        step: int | None = None,
        loss: float | None = None,
        dataset_path: str | None = None,
        batch_size: int | None = None,
        loss_history: list | None = None,
    ) -> None:
        """
        Save the model and optionally training state.

        Args:
            save_path: Path to save the model
            save_training_state: Whether to save training state (optimizer, scheduler, etc.)
            optimizer: Optimizer state to save
            scheduler: Scheduler state to save
            epoch: Current epoch
            step: Current step
            loss: Current loss
        """
        logger.setFunctionsName("save_model")
        if (
            not hasattr(config.hyper, "save_basepath")
            or config.hyper.save_basepath is None
        ):
            raise ValueError("Save path not configured in config.hyper.save_basepath")
        save_path = config.hyper.save_basepath

        new_version_dir = config.hyper.new_version_dir
        index_path = config.hyper.index_path
        fs = config.hyper.fs
        logger.step(f"Saving model to {save_path}...")
        import datetime

        os.makedirs(save_path, exist_ok=True)
        fs.append(
            {  # type: ignore
                "version": new_version_dir,
                "save_time": str(datetime.datetime.now()),
                "dataset_path": str(dataset_path),
                "model_path": save_path,
                "config": {
                    "base_model_name": model_name,
                    "use_lora": self.use_lora,
                    "loss": loss,
                    "num_outputs": self.num_outputs,
                    "max_length": config.dataset.max_length,
                    "learning_rate": optimizer.param_groups[0]["lr"]
                    if optimizer is not None
                    else None,
                    "epoch": epoch,
                    "step": step,
                    "batch_size": batch_size,
                },
            }
        )
        with open(index_path, "w+") as f:  # type: ignore
            json.dump(fs, f, indent=2)

        if self.use_lora:
            # Save LoRA adapters
            self.model.save_pretrained(save_path)
            logger.info(f"LoRA adapters saved to {save_path}")
        else:
            # Save full model
            self.model.save_pretrained(save_path)
            logger.info(f"Full model saved to {save_path}")

        # Save tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_path)  # type: ignore

        # Save numerical head if it exists
        if hasattr(self, "numerical_head") and hasattr(
            self.numerical_head, "state_dict"
        ):
            torch.save(
                self.numerical_head.state_dict(),  # type: ignore
                os.path.join(save_path, "numerical_head.pt"),
            )
            logger.info(f"Numerical head saved to {save_path}/numerical_head.pt")

        if save_training_state:
            training_state = {
                "epoch": epoch,
                "step": step,
                "loss": loss,
            }

            if optimizer is not None:
                training_state["optimizer_state_dict"] = optimizer.state_dict()

            if scheduler is not None:
                training_state["scheduler_state_dict"] = scheduler.state_dict()

            torch.save(training_state, os.path.join(save_path, "training_state.pt"))
            logger.info(f"Training state saved to {save_path}/training_state.pt")

        os.makedirs(os.path.join(save_path, "analysis"), exist_ok=True)
        if loss_history is not None:
            np.save(
                os.path.join(save_path, "analysis", "loss_history.npy"),
                np.array(loss_history),
            )
            logger.info(f"Loss history saved to {save_path}/loss_history.npy")

    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the model.

        Returns:
            Dictionary containing model information
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        # Add numerical head parameters if present
        if hasattr(self, "numerical_params") and hasattr(
            self.numerical_head, "parameters"
        ):
            numerical_params = sum(p.numel() for p in self.numerical_head.parameters())  # type: ignore
            total_params += numerical_params
            trainable_params += numerical_params
        else:
            numerical_params = 0

        return {
            "use_lora": self.use_lora,
            "num_outputs": self.num_outputs,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_ratio": trainable_params / total_params * 100,
            "numerical_head_parameters": numerical_params,
        }

    def train_mode(self, LLM_trainable: bool = True):
        logger.setFunctionsName("train_mode transition")
        logger.info(f"Setting model to train mode. LLM_trainable={LLM_trainable}")
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = LLM_trainable
        
        if hasattr(self, "numerical_head") and self.numerical_head is not None:
            self.numerical_head.train()

    def eval_mode(self):
        logger.setFunctionsName("eval_mode transition")
        logger.info(f"Setting model to eval mode.")
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        if hasattr(self, "numerical_head") and self.numerical_head is not None:
            self.numerical_head.eval()

    def params(self):
        return list(self.model.parameters()) + list(self.numerical_head.parameters())  # type: ignore

    def get_gpu_memory_usage(self):
        """Get current GPU memory usage for all devices."""
        return gpu_util()
