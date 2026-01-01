import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import copy
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from torch.utils.data import Dataset
from transformers.trainer_pt_utils import LabelSmoother
import transformers
from dataclasses import dataclass
from typing import Dict, Sequence, Optional

# =========================
# Global Config
# =========================

NUM_LOOPS = 5
BASE_MODEL = "~/models/Llama-2-7b-chat-hf"

USER_OUTPUT_ROOT = "~llm/ft_models/netflix/user_loop"
ITEM_OUTPUT_ROOT = "~llm/ft_models/netflix/item_loop"

MAX_SEQ_LENGTH = 2048
use_flash_attention = False

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

# =========================
# Quantization + LoRA
# =========================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

# =========================
# User-side Dataset
# =========================

def last_index(lst, value):
    return next((len(lst) - i - 1 for i, x in enumerate(lst[::-1]) if x != value), -1)

def safe_ids(ids, max_value, pad_id):
    return [i if i < max_value else pad_id for i in ids]

def mask_tokenize(item, tokenizer):
    input_ids, labels = [], []

    system = (
        "You are a recommendation system capable of predicting user-item "
        "interactions based on collaborative filtering."
    )
    system = B_SYS + system + E_SYS

    def encode_inst(text):
        return tokenizer.encode(f"{B_INST} {text} {E_INST}")

    ids = encode_inst(system + item["Input1"])
    input_ids += ids
    labels += [LabelSmoother.ignore_index] * len(ids)

    resp = tokenizer.encode(item["Response1"], add_special_tokens=False)
    input_ids += resp + [tokenizer.eos_token_id]
    labels += resp + [tokenizer.eos_token_id]

    ids = encode_inst(item["Input2"])
    input_ids += ids
    labels += [LabelSmoother.ignore_index] * len(ids)

    resp = tokenizer.encode(item["Response2"], add_special_tokens=False)
    input_ids += resp + [tokenizer.eos_token_id]
    labels += resp + [tokenizer.eos_token_id]

    trunc = last_index(labels, LabelSmoother.ignore_index) + 1
    input_ids, labels = input_ids[:trunc], labels[:trunc]

    return (
        torch.tensor(safe_ids(input_ids, tokenizer.vocab_size, tokenizer.pad_token_id)),
        torch.tensor(safe_ids(labels, tokenizer.vocab_size, LabelSmoother.ignore_index)),
    )

class UserDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids, labels = mask_tokenize(copy.deepcopy(self.data[idx]), self.tokenizer)
        return dict(input_ids=ids, labels=labels)

@dataclass
class UserCollator:
    tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, batch):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [b["input_ids"] for b in batch],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [b["labels"] for b in batch],
            batch_first=True,
            padding_value=LabelSmoother.ignore_index,
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

# =========================
# Item-side Formatter
# =========================

def format_item_instruction(sample):
    return f"""
<s>[INST]
Now you are a profile generator.
{sample['Input']}
Please generate 5 identities and 5 interests.
[/INST]
{sample['Response']}</s>
"""

# =========================
# Training Functions
# =========================

def load_model(model_id):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        use_cache=False,
        use_flash_attention_2=use_flash_attention,
        device_map="auto",
    )
    model.config.pretraining_tp = 1
    model = prepare_model_for_kbit_training(model)
    return get_peft_model(model, peft_config)

def load_tokenizer(model_id):
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.pad_token = tok.unk_token
    tok.padding_side = "right"
    return tok

def train_user_side(model_id, loop_id):
    print(f"\n[Loop {loop_id}] USER SIDE")
    out_dir = f"{USER_OUTPUT_ROOT}/loop_{loop_id}"

    tokenizer = load_tokenizer(model_id)
    model = load_model(model_id)

    raw = load_dataset(
        "csv",
        data_files="~sft_data/netflix/cf_instruction_data.csv",
        split="train",
    )
    dataset = UserDataset(raw, tokenizer)
    collator = UserCollator(tokenizer)

    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        learning_rate=5e-5,
        save_strategy="epoch",
        bf16=True,
        logging_steps=100,
    )

    trainer = transformers.Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model()

    return out_dir

def train_item_side(model_id, loop_id):
    print(f"\n[Loop {loop_id}] ITEM SIDE")
    out_dir = f"{ITEM_OUTPUT_ROOT}/loop_{loop_id}"

    tokenizer = load_tokenizer(model_id)
    model = load_model(model_id)

    dataset = load_from_disk("~sft_data/netflix/item_side_instruction_hf")

    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=5,
        per_device_train_batch_size=1,
        learning_rate=2e-4,
        save_strategy="epoch",
        bf16=True,
        logging_steps=50,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        formatting_func=format_item_instruction,
        packing=True,
        args=args,
    )
    trainer.train()
    trainer.save_model()

    return out_dir

# =========================
# 🔁 MAIN LOOP
# =========================

if __name__ == "__main__":
    current_model = BASE_MODEL

    for loop_id in range(1, NUM_LOOPS + 1):
        user_ckpt = train_user_side(current_model, loop_id)
        item_ckpt = train_item_side(user_ckpt, loop_id)
        current_model = item_ckpt

    print("\n✅ Training finished.")
