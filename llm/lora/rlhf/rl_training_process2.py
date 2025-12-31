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
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from peft import PeftModel
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, pipeline

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from trl.core import LengthSampler
from trl.import_utils import is_npu_available, is_xpu_available
from accelerate import Accelerator
import wandb
import os
os.environ["WANDB_MODE"] = "offline"
wandb.init()
input_min_text_length = 300
input_max_text_length = 2048


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    model_name: Optional[str] = field(default="~llm/ft_models/netflix/llama_lora_user_mask/checkpoint-5385", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(default="~sft_data/netflix/rlhf/rl.csv", metadata={"help": "the dataset name"})
    rm_adapter: Optional[str] = field(
        default="~llm/ft_models/netflix/rlhf/implicit_prm", metadata={"help": "the rm adapter name"}
    )
    log_with: Optional[str] = field(default='wandb', metadata={"help": "use 'wandb' to log with wandb"})
    # log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    use_safetensors: Optional[bool] = field(default=False, metadata={"help": "Use safetensors"})
    seed: Optional[int] = field(default=421, metadata={"help": "the random seed"})
    use_score_scaling: Optional[bool] = field(default=False, metadata={"help": "Use score scaling"})
    use_score_norm: Optional[bool] = field(
        default=False, metadata={"help": "Use score normalization. Only applicable if use_score_scaling is True"}
    )
    score_clip: Optional[float] = field(default=None, metadata={"help": "Score clipping"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

content_begin = "[INST] <<SYS>>\n You are a recommendation system capable of predicting user-item interactions based on the principles of collaborative filtering. Specifically, it can be devided into two stages. In the first stage, you will generate a user preference profile based on the user's historical behavior. In the second stage, using the preference profile generated in the first stage, you will find users with similar preferences and apply their historical interaction records to the target user. This allows you to determine whether the target user is likely to interact with a particular item in the future. \n<</SYS>>\n\n"
content_end = " strictly in the following format: User identity: [Identity 1], [Identity 2], [Identity 3]; User interests: [Interest 1], [Interest 2], [Interest 3]. Emphasize that only the most likely three identities and interests of the target user should be provided, and strictly adhere to the above format.[/INST]"

def create_and_prepare_dataset(tokenizer):
    dataset = load_dataset("csv", data_files=script_args.dataset_name, split="train")
    original_columns = dataset.column_names

    num_proc = 8

    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for question in examples["query"]:
            query = "Question: " + question + "\n\nAnswer: "
            query_mask_model = content_begin + question[:-1] + content_end
            # print(query_mask_model)
            tokenized_question = tokenizer(query_mask_model, truncation=True)
            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_question["input_ids"])

        return new_examples

    ds = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    ds = ds.filter(lambda x: len(x["input_ids"]) < input_max_text_length, batched=False)

    ds.set_format(type="torch")
    return ds

current_device = Accelerator().local_process_index

lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16
)
# model = AutoModelForCausalLMWithValueHead.from_pretrained(
#     script_args.model_name,
#     device_map={"": "xpu:0"} if is_xpu_available() else {"": "npu:0"} if is_npu_available else {"": 0},
#     peft_config=lora_config,
#     quantization_config=nf4_config,
#     reward_adapter=script_args.rm_adapter,
#     use_safetensors=script_args.use_safetensors,
# )
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    script_args.model_name,
    # device_map={"": "xpu:0"} if is_xpu_available() else {"": "npu:0"} if is_npu_available else {"": 0},
    device_map={"": current_device},
    peft_config=lora_config,
    quantization_config=nf4_config,
)
base_model_path = "~llm/ft_models/netflix/llama_lora_user_mask/checkpoint-5385"
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, model_max_length=4096)

tokenizer_rm = AutoTokenizer.from_pretrained(base_model_path, model_max_length=4096)
base_model = AutoModelForSequenceClassification.from_pretrained(base_model_path, device_map={"": current_device})
model_rm = PeftModel.from_pretrained(base_model, script_args.rm_adapter)
# tokenizer.pad_token = tokenizer.eos_token

dataset = create_and_prepare_dataset(tokenizer)

print(dataset)

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


config = PPOConfig(
    model_name=script_args.model_name,
    log_with=script_args.log_with,
    learning_rate=1e-5,
    batch_size=1,
    mini_batch_size=1,
    gradient_accumulation_steps=1,
    optimize_cuda_cache=True,
    seed=script_args.seed,
    use_score_scaling=script_args.use_score_scaling,
    use_score_norm=script_args.use_score_norm,
    score_clip=script_args.score_clip,
)

ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
)

sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=model_rm,
    device_map={"": current_device},
    model_kwargs={"load_in_4bit": True},
    tokenizer=tokenizer_rm, # mask model的tokenizer不一样
    return_token_type_ids=False,
)

generation_kwargs = {
    "top_k": 0.0,
    "top_p": 0.9,
    "do_sample": True,
    "temperature": 0.9,
    # "repetition_penalty": 1.3,
    "pad_token_id": tokenizer.pad_token_id,
    "max_new_tokens": 100,
}

sent_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 1,
    # "trunction": True,
}

reward_baseline = 0.0

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    # print(epoch)
    question_tensors_mask_model = batch["input_ids"]

    response_tensors = ppo_trainer.generate(
        question_tensors_mask_model,
        return_prompt=False,
        **generation_kwargs,
    )
    batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
    # Compute reward score
    texts = [q + r.strip() for q, r in zip(batch["query"], batch["response"])]

    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    rewards = [torch.tensor(output[0]["score"] - reward_baseline) for output in pipe_outputs]

    # print(batch["response"][0].strip())
    # print(texts[0])
    # print(rewards[0])

    # inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(ppo_trainer.accelerator.device)
    # raw_rewards = ppo_trainer.accelerator.unwrap_model(ppo_trainer.model).compute_reward_score(**inputs)
    # rewards = [raw_rewards[i, -1, 1] for i in range(len(raw_rewards))]  # take last token

    # Run PPO step
    stats = ppo_trainer.step(question_tensors_mask_model, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

    # ppo_trainer.save_pretrained()
    # ppo_trainer.save_pretrained(script_args.output_dir + f"step_{epoch}")
    if epoch and epoch % 500 == 0:
        ppo_trainer.save_pretrained('~llm/ft_models/rlhf/netflix_final_rlhf_testmaskv1_output_process_batch_4_fixed_' + f"step_{epoch}")
