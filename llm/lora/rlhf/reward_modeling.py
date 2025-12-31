# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["WANDB_DISABLED"] = "true"

from dataclasses import dataclass, field
from typing import Optional

import tyro
from accelerate import Accelerator
from datasets import load_dataset, load_metric
import evaluate
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, TrainerCallback

from trl import RewardConfig, RewardTrainer, is_xpu_available
import torch

import numpy as np

tqdm.pandas()


# @dataclass
# class ScriptArguments:
#     model_name: str = "./../../ft_models/netflix/llama_lora_user_base/merged_model_75"
#     """the model name"""
#     dataset_name_train: str = "./../../../sft_data/netflix/rlhf/train.csv"
#     dataset_name_eval: str = "./../../../sft_data/netflix/rlhf/eval.csv"
#     """the dataset name"""
#     dataset_text_field: str = "text"
#     """the text field of the dataset"""
#     load_in_8bit: bool = False
#     """load the model in 8 bits precision"""
#     load_in_4bit: bool = True
#     """load the model in 4 bits precision"""
#     eval_first_step: bool = False
#     reward_config: RewardConfig = field(
#         default_factory=lambda: RewardConfig(
#             output_dir = "./../../ft_models/netflix/rlhf/reward_model",
#             per_device_train_batch_size=4,
#             per_device_eval_batch_size=4,
#             num_train_epochs=3,
#             gradient_accumulation_steps=1,
#             gradient_checkpointing=True,
#             gradient_checkpointing_kwargs={"use_reentrant": False},
#             learning_rate=1.41e-5,
#             # report_to="tensorboard",
#             remove_unused_columns=False,
#             optim="adamw_torch",
#             logging_steps=500,
#             evaluation_strategy="steps",
#             eval_steps=500,
#             save_strategy="steps",
#             save_steps=500,
#             max_length=2048,
#             debug="underflow_overflow",
#             bf16=True,
#         )
#     )
#     use_peft: bool = True
#     """whether to use peft"""
#     peft_config: Optional[LoraConfig] = field(
#         default_factory=lambda: LoraConfig(
#             r=4,
#             lora_alpha=8,
#             bias="none",
#             task_type="SEQ_CLS",
#             modules_to_save=["scores"],
#             lora_dropout=0.1,
#         ),
#     )


# args = tyro.cli(ScriptArguments)

from dataclasses import dataclass, field
from typing import Optional
import tyro
from peft import LoraConfig
from trl import RewardConfig


@dataclass
class ScriptArguments:
    model_name: str = "~llm/ft_models/netflix/llama_lora_user_base/checkpoint-75"
    """The model name or path."""
    
    dataset_name_train: str = "./../../../sft_data/netflix/rlhf/train.csv"
    dataset_name_eval: str = "./../../../sft_data/netflix/rlhf/eval.csv"
    """Training and evaluation dataset paths."""

    dataset_text_field: str = "text"
    """Text field name in dataset."""

    load_in_8bit: bool = False
    load_in_4bit: bool = True

    eval_first_step: bool = False

    use_peft: bool = True

    # 为了兼容 tyro，暂时不将复杂结构作为默认值传入
    use_default_reward_config: bool = True
    """Whether to use the default RewardConfig."""

    use_default_peft_config: bool = True
    """Whether to use the default PEFT config."""


# 先解析简单参数
args = tyro.cli(ScriptArguments)

# 然后手动添加复杂对象
if args.use_default_reward_config:
    reward_config = RewardConfig(
        output_dir="~llm/ft_models/netflix/rlhf/reward_model",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=1.41e-5,
        remove_unused_columns=False,
        optim="adamw_torch",
        logging_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        max_length=2048,
        debug="underflow_overflow",
        bf16=True,
    )
else:
    reward_config = None  # 或你可以加逻辑读取自定义配置

args.reward_config = reward_config

if args.use_default_peft_config:
    peft_config = LoraConfig(
        r=4,
        lora_alpha=8,
        bias="none",
        task_type="SEQ_CLS",
        modules_to_save=["scores"],
        lora_dropout=0.1,
    )
else:
    peft_config = None
args.peft_config = peft_config
# 之后你可以继续使用这些参数
# 例如传入 RewardTrainer，或者打印验证：
print(f"model_name: {args.model_name}")
print(f"reward_config.bf16: {reward_config.bf16}")
print(f"peft_config.r: {peft_config.r if peft_config else 'None'}")


# Step 1: Load the model
if args.load_in_8bit and args.load_in_4bit:
    raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
elif args.load_in_8bit or args.load_in_4bit:
    quantization_config = BitsAndBytesConfig(load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    # quantization_config = BitsAndBytesConfig(load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    # Copy the model to each device
    device_map = (
        {"": f"xpu:{Accelerator().local_process_index}"}
        if is_xpu_available()
        else {"": Accelerator().local_process_index}
    )
else:
    device_map = None
    quantization_config = None

model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name,
    quantization_config=quantization_config,
    device_map=device_map,
    # trust_remote_code=args.trust_remote_code,
    num_labels=1,
    torch_dtype=torch.bfloat16,
    # torch_dtype=torch.float32,
)
# model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
model = prepare_model_for_kbit_training(model)

# Step 2: Load the dataset and pre-process it
tokenizer = AutoTokenizer.from_pretrained(args.model_name, model_max_length=4096)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
train_dataset = load_dataset("csv", data_files=args.dataset_name_train, split="train")
eval_dataset = load_dataset("csv", data_files=args.dataset_name_eval, split="train")

# train_dataset = train_dataset.select(range(10000))
eval_dataset = eval_dataset.select(range(1000))

# Tokenize chosen/rejected pairs of inputs
# Adapt this section to your needs for custom datasets
def preprocess_function(examples):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    # for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
    #     tokenized_chosen = tokenizer(chosen)
    #     tokenized_rejected = tokenizer(rejected)

    #     new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
    #     new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
    #     new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
    #     new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

    for query, response_c, response_r in zip(examples["query"], examples["chosen"], examples["rejected"]):
        tokenized_chosen = tokenizer("Question: " + query + "\n\nAnswer: " + response_c, truncation=True)
        tokenized_rejected = tokenizer("Question: " + query + "\n\nAnswer: " + response_r, truncation=True)

        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

    return new_examples

original_columns = train_dataset.column_names

# Preprocess the dataset and filter out examples that are longer than args.max_length
train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=original_columns,
)
train_dataset = train_dataset.filter(
    lambda x: len(x["input_ids_chosen"]) <= args.reward_config.max_length
    and len(x["input_ids_rejected"]) <= args.reward_config.max_length
)

eval_dataset = eval_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=original_columns,
)
eval_dataset = eval_dataset.filter(
    lambda x: len(x["input_ids_chosen"]) <= args.reward_config.max_length
    and len(x["input_ids_rejected"]) <= args.reward_config.max_length
)

print(train_dataset)
print(eval_dataset)

# Step 4: Define the LoraConfig
if args.use_peft:
    peft_config = args.peft_config
else:
    peft_config = None

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# model.config.torch_dtype=torch.float32
model.config.pad_token_id = model.config.eos_token_id

accuracy = load_metric('./accuracy.py', trust_remote_code=True)
#accuracy = evaluate.load('./accuracy_evaluate.py', trust_remote_code=True)

def compute_metrics(eval_pred):
    predictions, _ = eval_pred
    predictions = np.argmax(predictions, axis=0)
    labels = np.zeros(predictions.shape)
    return accuracy.compute(predictions=predictions, references=labels)

# Step 5: Define the Trainer
trainer = RewardTrainer(
    model=model,
    tokenizer=tokenizer,
    args=args.reward_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    # compute_metrics=compute_metrics,
)

if args.eval_first_step:

    class EvaluateFirstStepCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 1:
                control.should_evaluate = True
    
    trainer.add_callback(EvaluateFirstStepCallback())

trainer.train()
trainer.save_model()
# model.save_pretrained(args.reward_config.output_dir)