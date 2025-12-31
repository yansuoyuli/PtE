# coding=utf-8
# Copyright 2023 The HuggingFace Inc.
# Licensed under the Apache License, Version 2.0

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["WANDB_MODE"] = "offline"

from dataclasses import dataclass, field
from typing import Optional, List

import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig
from accelerate import Accelerator
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    HfArgumentParser,
    pipeline,
)
from trl import GRPOConfig, GRPOTrainer
from tqdm import tqdm

wandb.init()

# -------------------------
# 基本参数
# -------------------------
input_min_text_length = 300
input_max_text_length = 2048

accelerator = Accelerator()
current_device = accelerator.local_process_index


# -------------------------
# 参数定义
# -------------------------
@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(
        default="~llm/ft_models/netflix/llama_lora_user_mask/checkpoint-5385"
    )
    dataset_name: Optional[str] = field(
        default="~sft_data/netflix/rlhf/rl.csv"
    )
    rm_adapter: Optional[str] = field(
        default="~llm/ft_models/netflix/rlhf/reward_model/checkpoint-5000"
    )
    log_with: Optional[str] = field(default="wandb")
    seed: Optional[int] = field(default=421)


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


# -------------------------
# Prompt 模板
# -------------------------
content_begin = (
    "[INST] <<SYS>>\n"
    "You are a recommendation system capable of predicting user-item interactions "
    "based on the principles of collaborative filtering. Specifically, it can be "
    "divided into two stages. In the first stage, you will generate a user preference "
    "profile based on the user's historical behavior. In the second stage, using the "
    "preference profile generated in the first stage, you will find users with similar "
    "preferences and apply their historical interaction records to the target user.\n"
    "<</SYS>>\n\n"
)

content_end = (
    " strictly in the following format: "
    "User identity: [Identity 1], [Identity 2], [Identity 3]; "
    "User interests: [Interest 1], [Interest 2], [Interest 3]. "
    "Only output the final answer. [/INST]"
)


# -------------------------
# Dataset 构建
# -------------------------
def create_and_prepare_dataset(tokenizer):
    dataset = load_dataset("csv", data_files=script_args.dataset_name, split="train")
    original_columns = dataset.column_names

    def preprocess_function(examples):
        new_examples = {"prompt": []}
        for question in examples["query"]:
            prompt = content_begin + question[:-1] + content_end
            new_examples["prompt"].append(prompt)
        return new_examples

    dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=original_columns,
        num_proc=8,
    )

    dataset = dataset.filter(
        lambda x: len(tokenizer(x["prompt"])["input_ids"]) < input_max_text_length
    )

    return dataset


# -------------------------
# LoRA + 4bit 配置
# -------------------------
lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)


# -------------------------
# 模型 & Tokenizer
# -------------------------
model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    device_map={"": current_device},
    quantization_config=bnb_config,
    peft_config=lora_config,
)

tokenizer = AutoTokenizer.from_pretrained(
    script_args.model_name, model_max_length=4096
)
tokenizer.pad_token = tokenizer.eos_token


# Reward Model tokenizer
tokenizer_rm = AutoTokenizer.from_pretrained(
    script_args.rm_adapter, model_max_length=4096
)


# -------------------------
# 数据集
# -------------------------
dataset = create_and_prepare_dataset(tokenizer)
print(dataset)


# -------------------------
# Reward Model（情感/偏好打分）
# -------------------------
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=script_args.rm_adapter,
    tokenizer=tokenizer_rm,
    device_map={"": current_device},
    model_kwargs={"load_in_4bit": True},
    return_token_type_ids=False,
)


# -------------------------
# Reward Function（GRPO 核心）
# -------------------------
def reward_fn(prompts: List[str], completions: List[str]) -> List[float]:
    texts = [p + c for p, c in zip(prompts, completions)]

    outputs = sentiment_pipe(
        texts,
        return_all_scores=True,
        function_to_apply="none",
        batch_size=1,
    )

    rewards = [out[0]["score"] for out in outputs]
    return rewards


# -------------------------
# GRPO 配置
# -------------------------
grpo_config = GRPOConfig(
    model_name=script_args.model_name,
    log_with=script_args.log_with,
    learning_rate=1e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_generations=4,              # ⭐ GRPO 的 group size
    max_prompt_length=2048,
    max_completion_length=100,
    temperature=0.9,
    top_p=0.9,
    seed=script_args.seed,
)


# -------------------------
# GRPO Trainer
# -------------------------
trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    args=grpo_config,
    train_dataset=dataset,
)


# -------------------------
# 开始训练
# -------------------------
trainer.train(reward_fn=reward_fn)


# -------------------------
# 保存模型
# -------------------------
trainer.save_model(
    "~llm/ft_models/rlhf/netflix_grpo_final"
)
