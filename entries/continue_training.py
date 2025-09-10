from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
import json 
from torch.optim import AdamW
from tqdm import tqdm
from pathlib import Path
from typing import Optional, List, Dict, Any
from hypersurrogatemodel import (
    Logger, 
    TrainableLLM, 
)

from hypersurrogatemodel.config import config
logger = Logger("ContinueTraining")
logger.setFunctionsName("train")

class ModelWithCustomHead(TrainableLLM):
    def __init__(self, base_model, custom_head_path):
        super().__init__()
        self.base_model = base_model
        
        # Load head layer (weights and structure from file)
        custom_head_state = torch.load(custom_head_path, weights_only=True)
        hidden_size = base_model.config.hidden_size
        
        self.numerical_head = nn.Sequential(
            nn.Linear(hidden_size, custom_head_state['0.weight'].shape[0]),  # Layer 0
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(custom_head_state['0.weight'].shape[0], 1)  # Layer 3
        )
        self.numerical_head.load_state_dict(custom_head_state)
    
    def forward(self, input_ids, attention_mask=None, numerical_targets=None, **kwargs):
        # Get hidden states from base model
        outputs = self.base_model(input_ids, attention_mask=attention_mask, output_hidden_states=True, **kwargs)
        
        # Use last hidden state for numerical head
        last_hidden_state = outputs.hidden_states[-1]
        
        # Handle attention mask for getting last token
        if attention_mask is not None:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.size(0)
            last_token_hidden = last_hidden_state[range(batch_size), sequence_lengths]
        else:
            last_token_hidden = last_hidden_state[:, -1, :]
        
        # Apply numerical head
        numerical_output = self.numerical_head(last_token_hidden)
        
        result = {
            'logits': outputs.logits,
            'numerical_output': numerical_output,
            'hidden_states': outputs.hidden_states
        }
        
        # Calculate loss if targets provided
        if numerical_targets is not None:
            loss = nn.MSELoss()(numerical_output, numerical_targets)
            result['loss'] = loss
        
        return result

def continue_training(
    model_path="./saved_model/v1", 
    save_path="./saved_model",
    addition_name:Optional[str] =None,
    dataset_path="./data/processed/NAS_bench_201/cifar10_cleaned.json",
    epochs=config.training.num_epochs, 
    batch_size=config.training.batch_size, 
    learning_rate=config.training.learning_rate
):
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("Loading pretrained model...")
    
    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation="eager")
    
    # Create combined model with custom head
    model = ModelWithCustomHead(base_model, f"{model_path}/numerical_head.pt")
    
    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    
    logger.info(f"Model loaded and moved to {device}")
    
    # Load training data
    with open(dataset_path, 'r') as f:
        train_data = json.load(f)
    
    train_length = len(train_data)
    logger.info(f"loaded {train_length} training samples")
    
    batches_per_epoch = (len(train_data) + batch_size - 1) // batch_size
    total_batches = epochs * batches_per_epoch
    
    logger.info(f"total train of {total_batches} batches ({epochs} epochs × {batches_per_epoch} batches/epoch)")
    
    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Training progress bars
    epoch_pbar = tqdm(
        range(epochs), 
        desc="Epochs", 
        position=0
    )
    overall_pbar = tqdm(
        total=train_length*epochs,
        desc="Overall",
        position=1
    )
    
    
    global_batch_count = 0
    
    for epoch in epoch_pbar:
        total_loss = 0
        num_batches = 0
        
        batch_pbar = tqdm(
            range(0, len(train_data), batch_size), 
            desc="Batches", 
            position=2, 
            leave=False
        )
        
        for i in batch_pbar:
            batch_data = train_data[i:i+batch_size]
            texts = []
            targets = []
            
            for sample in batch_data:
                text = sample['text']
                answer = float(sample['answer'])
                texts.append(text)
                targets.append(answer)
            
            # Tokenize
            inputs = tokenizer(
                texts, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=512,
                return_attention_mask=True
            )
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            numerical_targets = torch.tensor(targets, device=device).unsqueeze(1)
            
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                numerical_targets=numerical_targets
            )
            
            loss = outputs.get('loss', None)
            if loss is None:
                logger.error("Failed to get loss")
                continue
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # clear cache
            if global_batch_count % 10 == 0:
                torch.cuda.empty_cache()
                
                
            # Track loss
            total_loss += loss.item()
            num_batches += 1
            global_batch_count += 1
            
            # Update progress bar
            avg_loss = total_loss / num_batches
            batch_pbar.set_postfix({
                'batch_loss': f'{avg_loss:.4f}',
            })
            overall_pbar.update(len(batch_data))
            overall_pbar.set_postfix({
                'Avg Loss': f'{avg_loss:6.3f}',
                'Epoch': f'{epoch+1}/{epochs}'
            })
        
        batch_pbar.close()
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        torch.cuda.empty_cache()
        
        # epoch progress bar 
        epoch_pbar.set_postfix({
            'Epoch Avg Loss': f'{avg_loss:.4f}',
        })
    
    # Save base model
    model.save_model(save_path=save_path, 
                    addtion_name=addition_name,
                    save_training_state=True,
                    optimizer=optimizer,
                    epoch=epochs,
                    dataset_path=dataset_path,
                    batch_size=batch_size,
                    )
    
    logger.success(f"Model saved successfully to {save_path}")
    
    return model

def test_continued_model(model_path="./saved_model_continued"):
    """Test the continued trained model"""
    logger.info("Testing continued model...")
    
    # Load continued model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_model = AutoModelForCausalLM.from_pretrained(model_path)
    model = ModelWithCustomHead(base_model, f"{model_path}/numerical_head.pt")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Test with sample data
    with open("./data/processed/NAS_bench_201/cifar10_cleaned.json", "r") as f:
        data = json.load(f)
    
    results = []
    for item in data[:3]:  # Test first 3 items
        input_ids = tokenizer(item['text'], return_tensors="pt").to(device)
        
        with torch.no_grad():
            prediction = model(**input_ids)
            result = prediction['numerical_output'].cpu().numpy().tolist()
            
            results.append({
                'input': item['text'],
                'prediction': result,
                'target': item['answer']
            })
            
            logger.info(f"Input: {item['text'][:50]}...")
            logger.info(f"Prediction: {result}, Target: {item['answer']}")
    
    return results

if __name__ == "__main__":
    # Continue training from saved model
    base_dir = Path(__file__).parent.parent
    continued_model = continue_training(
        model_path=str(base_dir / "saved_model/v1"),
        save_path=str(base_dir / "saved_model"),
        addition_name="continued",
        dataset_path=str(base_dir / "data/processed/NAS_bench_201/cifar100_cleaned.json"),
        epochs=config.training.num_epochs,
        batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate
    )
    
    # Test the continued model
    #test_results = test_continued_model("./saved_model_continued")
    
    logger.success("Continue training completed!")