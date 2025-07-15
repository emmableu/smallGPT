import os
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
import requests
import yaml
from torch.utils.data import Dataset
from dataclasses import dataclass
from transformers import Trainer, TrainingArguments
from model import GPT, GPTConfig

# Load GPTConfig from YAML
def load_gpt_config(yaml_path):
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return GPTConfig(**config_dict)

@dataclass
class TrainConfig:
    out_dir: str
    eval_interval: int
    eval_iters: int
    log_interval: int
    dataset: str
    gradient_accumulation_steps: int
    batch_size: int
    learning_rate: float
    max_iters: int
    lr_decay_iters: int
    min_lr: float
    beta2: float
    warmup_iters: int

    @staticmethod
    def from_yaml(path):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return TrainConfig(**data)

class CharDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+self.block_size], dtype=torch.long)
        y = torch.tensor(self.data[idx+1:idx+self.block_size+1], dtype=torch.long)
        return {"input_ids": x, "labels": y}

def main():
    # Load text
    input_file_path = os.path.join(PROJECT_ROOT, 'data/shakespeare/input.txt')
    if not os.path.exists(input_file_path):
        os.makedirs(os.path.dirname(input_file_path), exist_ok=True)
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with open(input_file_path, 'w', encoding='utf-8') as f:
            f.write(requests.get(data_url).text)

    with open(input_file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Tokenize
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = { ch:i for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s]
    data = torch.tensor(encode(text), dtype=torch.long)

    # Split
    train_data = data[:int(0.9*len(data))]
    val_data = data[int(0.9*len(data)):]

    # Load configs
    gpt_cfg = load_gpt_config(os.path.join(PROJECT_ROOT, "gpt_config/gpt_1m.yaml"))
    gpt_cfg.vocab_size = vocab_size
    model = GPT(gpt_cfg)

    train_cfg = TrainConfig.from_yaml(os.path.join(PROJECT_ROOT, "train_config/shakespeare_1m.yaml"))

    # Dataset
    train_dataset = CharDataset(train_data, gpt_cfg.block_size)
    eval_dataset = CharDataset(val_data, gpt_cfg.block_size)

    run_dir = os.path.join(train_cfg.out_dir, train_cfg.run_id)
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    logs_dir = os.path.join(run_dir, "logs")
    final_dir = os.path.join(run_dir, "final")

    hf_args = TrainingArguments(
        output_dir=checkpoints_dir,
        eval_strategy="steps",
        eval_steps=train_cfg.eval_interval,
        logging_steps=train_cfg.log_interval,
        per_device_train_batch_size=train_cfg.batch_size,
        per_device_eval_batch_size=train_cfg.batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        learning_rate=train_cfg.learning_rate,
        num_train_epochs=train_cfg.max_iters // len(train_dataset),
        weight_decay=0.1,
        warmup_steps=train_cfg.warmup_iters,
        save_strategy="no",  # Disable automatic checkpointing unless needed
        logging_dir=os.path.join(train_cfg.out_dir, "logs"),
        report_to="none",  # prevent wandb/etc if not set up
    )

    # Define trainer
    trainer = Trainer(
        model=model,
        args=hf_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    trainer.save_model(final_dir)

if __name__ == '__main__':
    main()