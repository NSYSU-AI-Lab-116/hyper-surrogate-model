import json 
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from hypersurrogatemodel.model import TrainingDataset, TrainableLLM, collate_fn
from hypersurrogatemodel.utils import Logger, get_gpu_utilization as gpu_u, get_device
from hypersurrogatemodel.config import config

logger = Logger(name="Pipelined-runner")
torch.set_float32_matmul_precision(precision='high')

if torch.cuda.is_available():
    torch.cuda.empty_cache()

def acc_trainer() -> None:
        """Enhanced training method using Accelerate for better performance and scalability"""
        
        # Load training data
        with open(config.dataset.train_data_path, 'r') as f:
            train_data = json.load(f)

        train_length = len(train_data)
        logger.info(f"Loaded {train_length} training samples")
        
        BATCH_SIZE = config.training.batch_size
        EPOCHS = config.training.num_epochs
        
        model = TrainableLLM(load_type="from_pretrained", use_lora=True)
        
        device = get_device(prefer_gpu=True)
        # Wrap the model with DataParallel
        if False:#torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            dp_model = torch.nn.DataParallel(model)
            dp_model = dp_model.to(device)
            model = dp_model.module  # Access the original model
        else:
            logger.info(f"Using single GPU")
            model = model.to(device)
        
        # Create dataset and dataloader
        train_dataset = TrainingDataset(train_data, model.tokenizer, max_length=512)
        train_dataloader = DataLoader(
            dataset=train_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )

        
        # Setup optimizer
        optimizer = AdamW(model.params(), lr=config.training.learning_rate)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
        
        num_update_steps_per_epoch = len(train_dataloader)
        max_train_steps = EPOCHS * num_update_steps_per_epoch
        
        
        
        # Log training info
        logger.info(f"***** Running training *****")
        logger.info(f"  Num examples = {train_length}")
        logger.info(f"  Num Epochs = {EPOCHS}")
        logger.info(f"  Batch size per device = {BATCH_SIZE}")
        logger.info(f"  Total optimization steps = {max_train_steps}")
        
        # Training progress tracking
        loss_history = []
        global_step = 0
        
        progress_bar = tqdm(
            range(max_train_steps), 
            desc="Epochs", 
            leave=True,
            position=1
        )
        gpu_util = tqdm(
            total=0,
            desc="Device util", 
            leave=True,
            position=0,
            bar_format='{desc}: {postfix}'
        )
        
        # Training strat
        for epoch in range(EPOCHS):
            model.train_mode()
            total_loss = 0
            batch_bar = tqdm(
                range(len(train_dataloader)),
                desc="Batches",
                leave=False,
                position=2
            )
            for step, batch in enumerate(train_dataloader):
                # Forward pass
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    numerical_targets=batch["numerical_targets"]
                )
                
                loss = outputs.get('loss')
                if loss is None:
                    logger.error(f"Warning: Loss is None at step {step}")
                    continue
                
                loss.backward()
                
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.detach().float()
                global_step += 1
                
                # Update progress bar
            
                batch_bar.update(1)
                progress_bar.update(1)
                avg_loss = total_loss / (step + 1)
                batch_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}'
                })
                loss_history.append(avg_loss.item())
                
                
                if global_step % 5 == 0:
                    gpu_util_percent = []
                    gpu_mem_percent = []
                    gpu_used_gb = []
                    gpu_total_gb = []
                    devices = gpu_u().get('devices', [])
                    for gpu in devices:
                        gpu_util_percent.append(gpu.get('gpu_utilization_percent', 0))
                        gpu_mem_percent.append(gpu.get('memory_utilization_percent', 0))
                        gpu_used_gb.append(gpu.get('memory_used_gb', 0))
                        gpu_total_gb.append(gpu.get('memory_total_gb', 0))
                    
                    gpu_util.set_postfix_str(
                        " // ".join(f"""GPU{i} Util: {gpu_util_percent[i]:.1f}% | GPU Mem: {gpu_mem_percent[i]:.1f}% ({gpu_used_gb[i]:.1f}GB/{gpu_total_gb[i]:.1f}GB)""" for i in range(len(devices)))
                    )
                # Clear cache periodically
                if global_step % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            batch_bar.close()
            
            epoch_avg_loss = total_loss / len(train_dataloader)
            progress_bar.set_postfix({
                'epoch': f'{epoch+1}/{EPOCHS}',
                'loss': f'{epoch_avg_loss:.4f}'
            })
            scheduler.step()
                
        progress_bar.close()
        model.eval_mode()
        model.save_model(
            model_name=model.model_path,
            save_training_state=True,
            optimizer=optimizer,
            epoch=EPOCHS,
            dataset_path=config.dataset.train_data_path,
            batch_size=BATCH_SIZE,
            loss_history=loss_history if loss_history else None
        )


if __name__ == "__main__":
    acc_trainer()