import json
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import warnings
warnings.filterwarnings(
    "ignore",
    message="No device id is provided",
    category=UserWarning,
    module=r"torch\.distributed\.distributed_c10d"
)
import time
import signal
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from hypersurrogatemodel.model import TrainingDataset, TrainableLLM, collate_fn
from hypersurrogatemodel.utils import Logger, print_gpu_utils as gpushow, setup_distributed
from hypersurrogatemodel.config import config
import torch.distributed as dist


logger = Logger(name="Pipelined-runner")
torch.set_float32_matmul_precision(precision="high")

STOP_REQUESTED = False
STEP_TO_SYNC = 10

def _signal_handler(signum, frame):
    global STOP_REQUESTED
    logger.info(f"Signal {signum} received — requesting stop")
    STOP_REQUESTED = True
    
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)
    
def load_training_data(data_path):
    logger.info(f"Loading training data from {data_path}")
    with open(data_path, 'r') as f:
        train_data = json.load(f)
    return train_data

def create_dataloaders(train_data, tokenizer, batch_size):
    train_dataset = TrainingDataset(train_data, tokenizer, max_length=512)
    sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    return train_dataset, train_dataloader, sampler

def initialize_model(device):
    """initialize model and wrap with DDP"""
    model = TrainableLLM(load_type="from_pretrained", use_lora=True)
    info = model.get_model_info()
    logger.info(f"Model initialized with {info['total_parameters']:,} parameters")
    logger.info(f"Trainable parameters: {info['trainable_parameters']:,}")
    model = model.to(device)
    
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs with DistributedDataParallel")
        dp_model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
            output_device=torch.cuda.current_device()
        )
        model = dp_model.module
    else:
        logger.info(f"Using single GPU")
    
    return model

def synchronize_all_processes():
    """Ensure all processes sync here."""
    if dist.is_initialized():
        dist.barrier()

def save_model_when_all_done(model, optimizer, epochs, data_path, batch_size, loss_history):
    """only save model from local rank"""
    synchronize_all_processes()
    
    if dist.get_rank() if dist.is_initialized() else 0 == 0:
        logger.info("All processes completed training. Saving model...")
        model.eval_mode()
        model.save_model(
            model_name=model.model_path,
            save_training_state=True,
            optimizer=optimizer,
            epoch=epochs,
            dataset_path=data_path,
            batch_size=batch_size,
            loss_history=loss_history if loss_history else None
        )
    
    synchronize_all_processes()

def trainer(loss_function) -> None:
    """train with distributed"""
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        try:
            setup_distributed()
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        except:
            logger.warning("Failed to initialize distributed training, falling back to single-GPU mode")
            distributed = False
            rank = 0
            world_size = 1
        
        BATCH_SIZE = config.training.batch_size
        EPOCHS = config.training.num_epochs
        
        train_data = load_training_data(config.dataset.train_data_path)
        train_length = len(train_data)
        if rank == 0:
            logger.info(f"Loaded {train_length} training samples")
        
        model = initialize_model(device)
        
        _, train_dataloader, sampler = create_dataloaders(
            train_data, model.tokenizer, BATCH_SIZE
        )
        
        optimizer = AdamW(model.params(), lr=config.training.learning_rate)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
        
        num_update_steps_per_epoch = len(train_dataloader)

        max_train_steps = EPOCHS * num_update_steps_per_epoch
        total_steps_all_gpus = max_train_steps * world_size

        if rank == 0:
            logger.info(f"***** Running training *****")
            logger.info(f"  Num examples = {train_length}")
            logger.info(f"  Num Epochs = {EPOCHS}")
            logger.info(f"  Batch size per device = {BATCH_SIZE}")
            logger.info(f"  Steps per GPU = {max_train_steps}")
            logger.info(f"  Total steps across all GPUs = {total_steps_all_gpus}")
            
            progress_bar = tqdm(
                range(total_steps_all_gpus), 
                desc=f"Overall Progress",
                leave=True,
                position=1
            )
            gpu_util = tqdm(
                total=0,
                desc=f"Rank {rank} (GPU {device.index}) Device util",
                leave=True,
                position=0,
                bar_format='{desc}: {postfix}'
            )
            
            
        synchronize_all_processes()
        
        # track
        loss_history = []
        global_step = 0
        
        # training
        for epoch in range(EPOCHS):
            if STOP_REQUESTED:
                break
            if sampler is not None:
                sampler.set_epoch(epoch)
                
            model.train_mode(LLM_trainable=False)
            total_loss = 0
        
            batch_bar = tqdm(
                range(len(train_dataloader)),
                desc=f"Rank {rank} (GPU {device.index}) Batches",
                leave=False,
                position=rank + 2
            )
            
            local_step_count = 0 
            
            for step, batch in enumerate(train_dataloader):
                if STOP_REQUESTED:
                    logger.info(f"Rank {rank} stopping early due to signal")
                    break
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
                    
                loss = loss.mean()
                loss.backward()
                
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.detach().float()
                global_step += 1
                local_step_count += 1
                
                batch_bar.update(1)
                avg_loss = total_loss / (step + 1)
                batch_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}'
                })
                
                if rank == 0:
                    loss_history.append(avg_loss.item())
                        
                # sync all gpu steps
                if step % STEP_TO_SYNC == 0:  # step to sync
                    step_tensor = torch.tensor(local_step_count, device=device)
                    torch.distributed.all_reduce(step_tensor, op=torch.distributed.ReduceOp.SUM)
                    
                    # update global progress bar
                    if rank == 0 and progress_bar is not None:
                        progress_bar.update(step_tensor.item())
                    
                    local_step_count = 0  # reset local step count

                if rank == 0 and global_step % 5 == 0 and gpu_util is not None:
                    gpushow(stream_obj=gpu_util)
                    gpu_util.refresh()
                
                if global_step % 20 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            batch_bar.close()
            
            epoch_avg_loss = total_loss / len(train_dataloader)
            if rank == 0 and progress_bar is not None:
                progress_bar.set_postfix({
                    'epoch': f'{epoch+1}/{EPOCHS}',
                    'loss': f'{epoch_avg_loss:.4f}'
                })
                
            scheduler.step()
            synchronize_all_processes()
        
        if rank == 0 and progress_bar is not None:
            progress_bar.close()
        
        save_model_when_all_done(
            model,
            optimizer,
            EPOCHS,
            config.dataset.train_data_path,
            BATCH_SIZE,
            loss_history
        )
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    trainer()