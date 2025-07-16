import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
import yaml
from dataclasses import dataclass, asdict
from transformers import Trainer, TrainingArguments
from model import GPT, GPTConfig
import argparse
import numpy as np


@dataclass
class TrainConfig:
    dataset: str
    eval_steps: int
    gradient_accumulation_steps: int
    logging_dir: str
    logging_steps: int
    learning_rate: float
    num_train_epochs: int
    out_dir: str
    per_device_eval_batch_size: int
    per_device_train_batch_size: int
    report_to: str
    run_id: str
    save_strategy: str
    warmup_steps: int
    weight_decay: float

    @staticmethod
    def from_yaml(path):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return TrainConfig(**data)

    def to_yaml(self, path):
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, sort_keys=False)

from torch.utils.data import Dataset
import torch

class TokenDataset(Dataset):
    """
    A PyTorch Dataset for loading tokenized sequences from a 1D array of token IDs.
    Each item is a pair of (input_ids, labels), where labels is input_ids shifted by one token.
    """
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.data[idx:idx + self.block_size], dtype=torch.long)
        labels = torch.tensor(self.data[idx + 1:idx + self.block_size + 1], dtype=torch.long)
        return {"inputs": input_ids, "labels": labels}

def run_train(gpt_config, train_config):

    # Load pre-tokenized .bin files
    bin_dir = os.path.join(PROJECT_ROOT, f'data/{train_config.dataset}')
    train_ids = np.memmap(os.path.join(bin_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_ids = np.memmap(os.path.join(bin_dir, 'val.bin'), dtype=np.uint16, mode='r')

    train_data = torch.tensor(train_ids[:], dtype=torch.long)
    val_data = torch.tensor(val_ids[:], dtype=torch.long)

    # Load configs
    model = GPT(gpt_config)

    run_dir = os.path.join(PROJECT_ROOT, train_cfg.out_dir, train_cfg.run_id)
    os.makedirs(run_dir, exist_ok=True)
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    logs_dir = os.path.join(run_dir, "logs")
    final_dir = os.path.join(run_dir, "final")

    train_config.to_yaml(run_dir + "/train_config.yaml")

    # Dataset
    train_dataset = TokenDataset(train_data, gpt_cfg.block_size)
    eval_dataset = TokenDataset(val_data, gpt_cfg.block_size)
    print(f"Trainer sees train dataset length: {len(train_dataset)}")
    print(f"Trainer sees train dataset length: {len(eval_dataset)}")

    hf_args = TrainingArguments(
        save_safetensors=False,
        output_dir=checkpoints_dir,
        eval_strategy="steps",
        eval_steps=train_config.eval_steps,
        logging_steps=train_config.logging_steps,
        per_device_train_batch_size=train_config.per_device_train_batch_size,
        per_device_eval_batch_size=train_config.per_device_eval_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        learning_rate=train_config.learning_rate,
        num_train_epochs=train_config.num_train_epochs,
        weight_decay=train_config.weight_decay,
        warmup_steps=train_config.warmup_steps,
        save_strategy=train_config.save_strategy,
        logging_dir=train_config.logging_dir,
        report_to=train_config.report_to,

    )

    trainer = Trainer(
        model=model,
        args=hf_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    trainer.save_model(final_dir)
    eval_metrics = trainer.evaluate()
    print(f"\n📉 Eval loss: {eval_metrics=}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt_config", type=str, required=True)
    parser.add_argument("--train_config", type=str, required=True)
    gpt_cfg = GPTConfig.from_yaml(os.path.join(PROJECT_ROOT, "gpt_config/gpt_1m.yaml"))
    train_cfg = TrainConfig.from_yaml(os.path.join(PROJECT_ROOT, "train_config/shakespeare_1m.yaml"))
    print(f'{gpt_cfg=}')
    print(f'{train_cfg=}')
    run_train(gpt_cfg, train_cfg)