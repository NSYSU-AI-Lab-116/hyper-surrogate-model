from pathlib import Path
import os
import json
import pandas as pd
from requests import get
import torch
from torch.optim import AdamW
from tqdm import tqdm
from hypersurrogatemodel.model import collate_fn, TrainingDataset, TrainableLLM
from hypersurrogatemodel.utils import Logger, get_device, get_gpu_utilization as gpu_u
from hypersurrogatemodel.config import config
import numpy as np
import random
from sklearn.model_selection import train_test_split

logger = Logger("Pipelined-runner")
torch.set_float32_matmul_precision("high")


def train_with_dataset(
    dataset_path,
    model_path="./saved_model",
    epochs=6,
    batch_size=12,
    learning_rate=1e-5,
):
    with open(dataset_path, "r") as f:
        train_data = json.load(f)

    train_length = len(train_data)
    logger.info(f"loaded {train_length} training samples")

    logger.info("Sorting training data by accuracy to prepare for ranking loss...")
    train_data.sort(key=lambda x: float(x["true_acc"]), reverse=True)
    logger.success(f"Sorted {len(train_data)} training samples.")

    batches_per_epoch = (len(train_data) + batch_size - 1) // batch_size
    total_batches = epochs * batches_per_epoch

    logger.info(
        f"total train of {total_batches} batches ({epochs} epochs × {batches_per_epoch} batches/epoch)"
    )

    model = TrainableLLM(load_type="from_pretrained")
    device = get_device(prefer_gpu=True)
    if False:  # torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        dp_model = torch.nn.DataParallel(model)
        dp_model = dp_model.to(device)
        model = dp_model.module  # Access the original model
    else:
        logger.info(f"Using single GPU")
        model = model.to(device)
    train_dataset = TrainingDataset(train_data, model.tokenizer, max_length=512)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    model.train_mode()

    # ranking loss
    loss_fn = torch.nn.MarginRankingLoss(margin=0.1)
    logger.info(f"Using MarginRankingLoss with a margin of {loss_fn.margin}")

    logger.info(f"開始訓練 {epochs} 個 epochs，批次大小: {batch_size}")

    num_pairs_per_epoch = len(train_data)
    num_batches_per_epoch = (num_pairs_per_epoch + batch_size - 1) // batch_size

    # Create overall progress bar
    total_pbar = tqdm(
        total=num_pairs_per_epoch * epochs,
        desc=f"{'Overall':8}",
        unit="pairs",
        position=1,
        leave=True,
    )
    gpu_util = tqdm(
        total=0,
        desc="Device util",
        leave=True,
        position=0,
        bar_format="{desc}: {postfix}",
    )

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        # create batch progress bar
        batch_pbar = tqdm(
            range(num_batches_per_epoch),
            desc=f"{'Batch':8}",
            unit="batch",
            position=2,
            leave=False,
        )

        for i in batch_pbar:
            good_prompts = []
            bad_prompts = []
            optimizer.zero_grad()

            # matching
            current_batch_size = min(
                batch_size, num_pairs_per_epoch - (num_batches * batch_size)
            )
            if current_batch_size <= 0:
                continue

            for _ in range(current_batch_size):  # 取得 current batch size 的配對數量
                # 四分位數
                q1 = len(train_data) // 4
                q2 = len(train_data) // 2
                q3 = q1 * 3

                # valid interval
                if q1 < 2 or (len(train_data) - q3) < 2:
                    good_sample = random.choice(train_data[:q2])
                    bad_sample = random.choice(train_data[q2:])
                    good_prompts.append(good_sample)
                    bad_prompts.append(bad_sample)
                    continue

                rand_val = random.random()

                # hard pairwise
                if rand_val < 0.4:
                    idx1 = random.randint(0, q1 - 2)
                    idx2 = random.randint(idx1 + 1, q1 - 1)
                    good_sample = train_data[idx1]
                    bad_sample = train_data[idx2]

                # hard pairwise for weak arch.
                elif rand_val < 0.7:
                    idx1 = random.randint(q3, len(train_data) - 2)
                    idx2 = random.randint(idx1 + 1, len(train_data) - 1)
                    good_sample = train_data[idx1]
                    bad_sample = train_data[idx2]

                # simple pairwise
                else:
                    good_sample = random.choice(train_data[:q2])
                    bad_sample = random.choice(train_data[q2:])

                good_prompts.append(good_sample)
                bad_prompts.append(bad_sample)

            if not good_prompts:
                continue
            good_inputs = TrainingDataset(good_prompts, model.tokenizer, max_length=512)
            bad_inputs = TrainingDataset(bad_prompts, model.tokenizer, max_length=512)
            good_inputs = collate_fn(good_inputs)
            bad_inputs = collate_fn(bad_inputs)
            good_inputs = {k: v.to(device) for k, v in good_inputs.items()}
            bad_inputs = {k: v.to(device) for k, v in bad_inputs.items()}

            optimizer.zero_grad()

            good_scores = model(
                input_ids=good_inputs["input_ids"],
                attention_mask=good_inputs["attention_mask"],
                numerical_targets=good_inputs["numerical_targets"],
            )["numerical_outputs"]
            bad_scores = model(
                input_ids=bad_inputs["input_ids"],
                attention_mask=bad_inputs["attention_mask"],
                numerical_targets=bad_inputs["numerical_targets"],
            )["numerical_outputs"]

            # compute Ranking Loss
            target = torch.ones_like(good_scores)
            loss = loss_fn(good_scores, bad_scores, target)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Overall progress updated
            total_pbar.update(len(good_prompts))
            total_pbar.set_postfix(
                {
                    "Avg Loss": f"{(total_loss / num_batches):.4f}",
                    "Epoch": f"{epoch + 1}/{epochs}",
                }
            )

            if total_batches % 5 == 0:
                gpu_util_percent = []
                gpu_mem_percent = []
                gpu_used_gb = []
                gpu_total_gb = []
                devices = gpu_u().get("devices", [])
                for gpu in devices:
                    gpu_util_percent.append(gpu.get("gpu_utilization_percent", 0))
                    gpu_mem_percent.append(gpu.get("memory_utilization_percent", 0))
                    gpu_used_gb.append(gpu.get("memory_used_gb", 0))
                    gpu_total_gb.append(gpu.get("memory_total_gb", 0))

                gpu_util.set_postfix_str(
                    " // ".join(
                        f"""GPU{i} Util: {gpu_util_percent[i]:.1f}% | GPU Mem: {gpu_mem_percent[i]:.1f}% ({gpu_used_gb[i]:.1f}GB/{gpu_total_gb[i]:.1f}GB)"""
                        for i in range(len(devices))
                    )
                )

            # Batch progress
            batch_pbar.set_postfix(
                {"Ranking Loss": f"{(total_loss / num_batches):.4f}"}
            )

        batch_pbar.close()

        avg_epoch_loss = total_loss / num_batches if num_batches > 0 else 0

    total_pbar.close()
    gpu_util.close()
    model.eval_mode()
    model.save_model(
        model_name=model.model_path,
        save_training_state=True,
        optimizer=optimizer,
        epoch=epochs,
        dataset_path=config.dataset.train_data_path,
        batch_size=batch_size,
    )

    logger.success("Finish training")
    return model


if __name__ == "__main__":
    train_with_dataset(
        dataset_path=Path("./data/processed/NAS_bench_201_train/cifar10_train.json"),
        epochs=config.training.num_epochs,
        batch_size=config.training.batch_size // 2,
        learning_rate=config.training.learning_rate,
    )
