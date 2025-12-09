from omegaconf import OmegaConf
from dataclasses import dataclass
from typing import Optional
from transformers import AutoTokenizer
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
import torch
from rich.console import Console
console = Console()

MODEL_PATH = r"/home/xiaodongqu/hyx_exp/LLMs/Qwen/Qwen3-8B"
DATA_PATH = r"/home/xiaodongqu/hyx_exp/verl-main/training_checklist/data/demo_multiturn.parquet"
MAX_SAMPLES = 3


def visualize_dataset_mask():
    print(f"Loading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code = True)

    config = OmegaConf.create(
        {
            "pad_mode": "right",
            "max_length": 5120,
            "truncation": "error",
            "multiturn": {
                "enable": True,
                "messages_key": "messages",
                "tools_key": "tools"
            }
        }
    )

    print(f"Loading dataset from {DATA_PATH}...")

    dataset = MultiTurnSFTDataset(
        parquet_files = [DATA_PATH],
        tokenizer = tokenizer,
        config = config,
        max_samples = MAX_SAMPLES
    )

    print(f"\n{'='*20} Checking {len(dataset)} Samples {'='*20}\n")

    for i in range(len(dataset)):
        item = dataset[i]
        input_ids = item["input_ids"]
        loss_mask = item["loss_mask"]

        # 确保是 tensor 并且在 CPU 上
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        if isinstance(loss_mask, torch.Tensor):
            loss_mask = loss_mask.tolist()

        # 跳过padding
        output_tokens = []
        for tid, mask in zip(input_ids, loss_mask):
            if tid == tokenizer.pad_token_id:
                continue
             
            token_text = tokenizer.decode([tid], skip_special_tokens=False)

            if mask == 1:
                output_tokens.append(f"[green]{token_text}[/green]")
            else:
                output_tokens.append(f"[grey62]{token_text}[/grey62]")
                
        console.print("".join(output_tokens))

if __name__ == "__main__":
    visualize_dataset_mask()

