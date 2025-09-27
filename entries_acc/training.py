import json
import torch
import os
import time
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

if torch.cuda.is_available():
    torch.cuda.empty_cache()

def load_training_data(data_path):
    """集中加載訓練數據的邏輯"""
    logger.info(f"Loading training data from {data_path}")
    with open(data_path, 'r') as f:
        train_data = json.load(f)
    return train_data

def create_dataloaders(train_data, tokenizer, batch_size, distributed=False):
    """創建數據加載器，處理分散式和非分散式情況"""
    train_dataset = TrainingDataset(train_data, tokenizer, max_length=512)
    sampler = DistributedSampler(train_dataset) if distributed else None
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

def initialize_model(device, distributed=False):
    """初始化模型，處理分散式和非分散式情況"""
    model = TrainableLLM(load_type="from_pretrained", use_lora=True)
    model = model.to(device)
    
    # 不要在這裡呼叫 setup_distributed()
    if distributed and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs with DistributedDataParallel")
        dp_model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[int(os.environ["LOCAL_RANK"])],
            output_device=int(os.environ["LOCAL_RANK"])
        )
        model = dp_model.module
    else:
        logger.info(f"Using single GPU")
    
    return model

def synchronize_all_processes():
    """確保所有進程同步到此點"""
    if dist.is_initialized():
        dist.barrier()

def save_model_when_all_done(model, optimizer, epochs, data_path, batch_size, loss_history):
    """在所有進程完成訓練後，僅由 rank=0 進程保存模型"""
    # 確保所有進程都完成訓練
    synchronize_all_processes()
    
    # 僅在主進程（rank=0）上保存模型
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
    
    # 再次同步，確保模型保存完成
    synchronize_all_processes()

def acc_trainer(distributed: bool = False) -> None:
    """重構後的分散式訓練函數"""
    try:
        # 設置設備
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}")
        else:
            device = torch.device("cpu")
            distributed = False  # CPU 模式下禁用分散式訓練
        
        # 獲取 rank 和 world_size
        if distributed:
            try:
                setup_distributed()
                rank = dist.get_rank()
                world_size = dist.get_world_size()
            except:
                logger.warning("Failed to initialize distributed training, falling back to single-GPU mode")
                distributed = False
                rank = 0
                world_size = 1
        else:
            rank = 0
            world_size = 1
        
        # 加載訓練參數 - 只顯示在 rank 0
        BATCH_SIZE = config.training.batch_size
        EPOCHS = config.training.num_epochs
        
        # 加載訓練數據 - 每個進程都需要，但只在 rank 0 記錄
        train_data = load_training_data(config.dataset.train_data_path)
        train_length = len(train_data)
        if rank == 0:
            logger.info(f"Loaded {train_length} training samples")
        
        # 初始化模型 - 對每個進程的模型單獨初始化
        model = initialize_model(device, distributed)
        
        # 創建數據加載器 - 分散式情況下使用 DistributedSampler
        _, train_dataloader, sampler = create_dataloaders(
            train_data, model.tokenizer, BATCH_SIZE, distributed
        )
        
        # 設置優化器
        optimizer = AdamW(model.params(), lr=config.training.learning_rate)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
        
        # 計算訓練步數
        num_update_steps_per_epoch = len(train_dataloader)

        if distributed:
            # 分散式訓練：總步數 = 每個 GPU 的步數 × GPU 數量
            # 使用 world_size 表示參與訓練的 GPU 數量
            max_train_steps = EPOCHS * num_update_steps_per_epoch
            total_steps_all_gpus = max_train_steps * world_size
        else:
            # 非分散式訓練：總步數就是單 GPU 的步數
            max_train_steps = EPOCHS * num_update_steps_per_epoch
            total_steps_all_gpus = max_train_steps

        # 只在主進程上創建進度條，使用正確的總步數
        if rank == 0:
            # 記錄訓練信息
            logger.info(f"***** Running training *****")
            logger.info(f"  Num examples = {train_length}")
            logger.info(f"  Num Epochs = {EPOCHS}")
            logger.info(f"  Batch size per device = {BATCH_SIZE}")
            logger.info(f"  Steps per GPU = {max_train_steps}")
            logger.info(f"  Total steps across all GPUs = {total_steps_all_gpus}")
            
            progress_bar = tqdm(
                range(total_steps_all_gpus),  # 使用所有 GPU 的總步數
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
            
            
        
        # 確保所有進程同步到此點
        synchronize_all_processes()
        
        # 訓練跟踪
        loss_history = []
        global_step = 0
        
        # 訓練開始
        for epoch in range(EPOCHS):
            # 設置 epoch 到採樣器（對分散式訓練很重要）
            if sampler is not None:
                sampler.set_epoch(epoch)
                
            model.train_mode()
            total_loss = 0
            
            # 在每個進程上創建批次進度條
            batch_bar = tqdm(
                range(len(train_dataloader)),
                desc=f"Rank {rank} (GPU {device.index}) Batches",
                leave=False,
                position=rank + 2  # 確保每個進程的進度條不重疊
            )
            
            local_step_count = 0  # 跟踪此 rank 的步數
            
            for step, batch in enumerate(train_dataloader):
                # 前向傳播
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
                
                # 更新本地進度條
                batch_bar.update(1)
                avg_loss = total_loss / (step + 1)
                batch_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}'
                })
                
                if rank == 0:
                    loss_history.append(avg_loss.item())
                
                # 更新主進度條的邏輯
                if rank == 0 and progress_bar is not None:
                    # 非分散式訓練，直接更新
                    if not distributed:
                        progress_bar.update(1)
                        
                # 分散式訓練的進度同步        
                if step % 5 == 0 and distributed:  # 每5步同步一次
                    step_tensor = torch.tensor(local_step_count, device=device)
                    # 所有 GPU 的步數相加
                    torch.distributed.all_reduce(step_tensor, op=torch.distributed.ReduceOp.SUM)
                    
                    # 在主進程上更新進度條
                    if rank == 0 and progress_bar is not None:
                        progress_bar.update(step_tensor.item())  # 更新主進度條
                    
                    local_step_count = 0  # 重置本地步數計數

                # 定期更新 GPU 利用率 (不依賴於分散式邏輯)
                if rank == 0 and global_step % 5 == 0 and gpu_util is not None:
                    gpushow(stream_obj=gpu_util)
                    gpu_util.refresh()
                
                # 定期清理 GPU 緩存
                if global_step % 20 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            # 完成一個批次
            batch_bar.close()
            
            # 計算 epoch 平均損失
            epoch_avg_loss = total_loss / len(train_dataloader)
            if rank == 0 and progress_bar is not None:
                progress_bar.set_postfix({
                    'epoch': f'{epoch+1}/{EPOCHS}',
                    'loss': f'{epoch_avg_loss:.4f}'
                })
                
            scheduler.step()
            
            # 確保所有進程完成當前 epoch
            synchronize_all_processes()
        
        # 訓練完成，關閉進度條
        if rank == 0 and progress_bar is not None:
            progress_bar.close()
        
        # 在所有進程完成訓練後，僅由主進程保存模型
        save_model_when_all_done(
            model,
            optimizer,
            EPOCHS,
            config.dataset.train_data_path,
            BATCH_SIZE,
            loss_history
        )
        
    finally:
        # 清理分散式進程組
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    acc_trainer(distributed=True)