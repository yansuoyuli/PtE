

import os
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import tyro
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model  # 新增 prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from trl import RewardConfig, RewardTrainer, is_xpu_available

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


# Custom imports for reference model and process reward calculation
from transformers.generation.utils import LogitsProcessorList
from torch.nn.functional import log_softmax

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["WANDB_DISABLED"] = "true"


@dataclass
class ScriptArguments:
    model_name: str = "~llm/ft_models/netflix/llama_lora_user_base/checkpoint-75"
    """The policy model name or path."""
    
    ref_model_name: str = "~/models/Vicuna_13b_v1.3"
    """The reference model name or path."""
    
    dataset_name_train: str = "./../../../sft_data/netflix/rlhf/train.csv"
    dataset_name_eval: str = "./../../../sft_data/netflix/rlhf/eval.csv"
    """Training and evaluation dataset paths."""

    dataset_text_field: str = "text"
    """Text field name in dataset."""

    load_in_8bit: bool = False
    load_in_4bit: bool = True

    eval_first_step: bool = False

    use_peft: bool = True

    beta: float = 0.05
    """Hyperparameter for reward scaling (β in the paper)."""

    use_default_reward_config: bool = True
    """Whether to use the default RewardConfig."""

    use_default_peft_config: bool = True
    """Whether to use the default PEFT config."""


# Parse arguments and initialize complex configurations
args = tyro.cli(ScriptArguments)

# Initialize RewardConfig with process reward-specific parameters
if args.use_default_reward_config:
    reward_config = RewardConfig(
        output_dir="~llm/ft_models/netflix/rlhf/implicit_prm2",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
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
    reward_config = None
args.reward_config = reward_config

# PEFT configuration for policy model
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


# Step 1: Load policy model (for reward calculation)
if args.load_in_8bit and args.load_in_4bit:
    raise ValueError("Cannot load model in 8-bit and 4-bit simultaneously")
quantization_config = BitsAndBytesConfig(
    load_in_8bit=args.load_in_8bit,
    load_in_4bit=args.load_in_4bit,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)
device_map = (
    {"": f"xpu:{Accelerator().local_process_index}"}
    if is_xpu_available()
    else {"": Accelerator().local_process_index}
)

policy_model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    quantization_config=quantization_config,
    device_map=device_map,
    torch_dtype=torch.bfloat16,
    trust_remote_code=False
)
policy_model = prepare_model_for_kbit_training(policy_model)

# Step 2: Load reference model
ref_model = AutoModelForCausalLM.from_pretrained(
    args.ref_model_name,
    quantization_config=quantization_config,
    device_map=device_map,
    torch_dtype=torch.bfloat16,
    trust_remote_code=False
)
ref_model = prepare_model_for_kbit_training(ref_model) if args.use_peft else ref_model

# Ensure both models are in evaluation mode
policy_model.eval()
ref_model.eval()

# Step 3: Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name, model_max_length=4096)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Step 4: Load and preprocess dataset
train_dataset = load_dataset("csv", data_files=args.dataset_name_train, split="train")
eval_dataset = load_dataset("csv", data_files=args.dataset_name_eval, split="train")

# Limit dataset size for testing (optional)
# train_dataset = train_dataset.select(range(10000))
eval_dataset = eval_dataset.select(range(1000))

def preprocess_function(examples):
    """Preprocess data to format queries and responses for process reward calculation."""
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
        "query": []
    }
    for query, response_c, response_r in zip(examples["query"], examples["chosen"], examples["rejected"]):
        # Format input with query and response
        input_chosen = f"Question: {query}\n\nAnswer: {response_c}"
        input_rejected = f"Question: {query}\n\nAnswer: {response_r}"
        
        tokenized_chosen = tokenizer(input_chosen, truncation=True)
        tokenized_rejected = tokenizer(input_rejected, truncation=True)
        
        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])
        new_examples["query"].append(query)
    return new_examples

original_columns = train_dataset.column_names

# Apply preprocessing and filter by length
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

# Step 5: Define custom reward model that computes process reward
class ImplicitPRM(torch.nn.Module):
    """Custom model to compute free process reward as log-likelihood ratio."""
    def __init__(self, policy_model, ref_model, tokenizer, beta=0.05):
        super().__init__()
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.beta = beta
        self.vocab_size = policy_model.config.vocab_size
    
    @property
    def config(self):
        return self.policy_model.config
    
    @property
    def device(self):
        return self.policy_model.device

    @property
    def dtype(self):
        return self.policy_model.dtype
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.policy_model, "gradient_checkpointing_enable"):
            self.policy_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        else:
            raise NotImplementedError(
                f"Model {type(self.policy_model).__name__} does not support gradient_checkpointing_enable."
            )
    def forward(self, input_ids, attention_mask,**kwargs):
        # if "inputs_embeds" in kwargs:
        #     inputs_embeds = kwargs["inputs_embeds"]
        # else:
        #     inputs_embeds = input_ids.new_zeros(
        #         (input_ids.shape[0], input_ids.shape[1]), 
        #         dtype=torch.long
        #     )  # 虚拟 input_ids（仅用于长度判断，不影响计算）

        # Compute log probabilities for policy model
        #policy_outputs = self.policy_model(input_ids, attention_mask=attention_mask,**kwargs)
        policy_outputs = self.policy_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            #inputs_embeds=inputs_embeds,
            #return_dict=True,
            **kwargs
        )
        policy_logits = policy_outputs.logits[:, :-1, :]  # Skip last token for next-token prediction
        #policy_log_probs = log_softmax(policy_logits, dim=-1)
        policy_lp = log_softmax(policy_logits, dim=-1)
        # Compute log probabilities for reference model
        #ref_outputs = self.ref_model(input_ids, attention_mask=attention_mask,**kwargs)
        ref_outputs = self.ref_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            #inputs_embeds=inputs_embeds,
            #return_dict=True,
            **kwargs
        )
        ref_logits = ref_outputs.logits[:, :-1, :]
        # ref_log_probs = log_softmax(ref_logits, dim=-1)
        
        # # Get target tokens (next tokens in the sequence)
        # target_ids = input_ids[:, 1:].unsqueeze(-1)  # Shifted input_ids
        
        # # Calculate log-likelihood ratio for each step
        # policy_target_probs = policy_log_probs.gather(2, target_ids).squeeze(-1)
        # ref_target_probs = ref_log_probs.gather(2, target_ids).squeeze(-1)
        # log_ratio = policy_target_probs - ref_target_probs
        
        # # Sum over steps to get total process reward (equivalent to outcome reward)
        # process_reward = self.beta * log_ratio.sum(dim=1, keepdim=True)
        # return {"rewards": process_reward}
        ref_lp = log_softmax(ref_logits, dim=-1)

        # 3) 计算 process_reward
        target_ids = input_ids[:, 1:].unsqueeze(-1)
        policy_p = policy_lp.gather(2, target_ids).squeeze(-1)
        ref_p    = ref_lp.gather(2, target_ids).squeeze(-1)
        log_ratio = policy_p - ref_p
        process_reward = self.beta * log_ratio.sum(dim=1, keepdim=True)

        # 4) 返回带 logits 和 rewards 的 ModelOutput
        # return SequenceClassifierOutput(
        #     logits=process_reward,    # 让 Trainer 找到 “logits”
        #     loss=None,                # 你真正的 loss 会在 compute_loss 里算
        #     hidden_states=None,
        #     attentions=None,
        #     # 自定义字段
        #     **{"rewards": process_reward}
        # )
        #return {"rewards": process_reward, "logits": process_reward}
        return {
            "logits": {"process_reward": process_reward},
            "rewards": process_reward,
        }
# Wrap the model for RewardTrainer
class PRMTrainer(RewardTrainer):
    """Custom RewardTrainer to handle process reward calculation."""
    # def compute_rewards(self, batch):
    #     # Unpack batch
    #     input_ids_chosen = batch["input_ids_chosen"]
    #     attention_mask_chosen = batch["attention_mask_chosen"]
    #     input_ids_rejected = batch["input_ids_rejected"]
    #     attention_mask_rejected = batch["attention_mask_rejected"]
        
    #     # Compute rewards for chosen and rejected responses
    #     chosen_rewards = self.model(input_ids_chosen, attention_mask_chosen)["rewards"]
    #     rejected_rewards = self.model(input_ids_rejected, attention_mask_rejected)["rewards"]
        
    #     # Return rewards in the format expected by RewardTrainer
    #     return {
    #         "rewards_chosen": chosen_rewards,
    #         "rewards_rejected": rejected_rewards
    #     }
    # 提取 chosen 和 rejected 的输入字段（确保键名与预处理后的数据集一致）
    # def compute_rewards(self, batch):    
    #     chosen_inputs = {
    #         "input_ids": batch["input_ids_chosen"],
    #         "attention_mask": batch["attention_mask_chosen"]
    #     }
    #     rejected_inputs = {
    #         "input_ids": batch["input_ids_rejected"],
    #         "attention_mask": batch["attention_mask_rejected"]
    #     }
        
    #     # 调用模型时使用 ** 解包字典参数
    #     chosen_rewards = self.model(**chosen_inputs)["rewards"]
    #     rejected_rewards = self.model(**rejected_inputs)["rewards"]
        
    #     return {
    #         "rewards_chosen": chosen_rewards,
    #         "rewards_rejected": rejected_rewards
    #     }
    # def compute_loss(self, model, inputs, return_outputs=False):
    #     # Unpack chosen and rejected inputs from the batch dict
    #     chosen_inputs = {
    #         "input_ids": inputs["input_ids_chosen"],
    #         "attention_mask": inputs["attention_mask_chosen"]
    #     }
    #     rejected_inputs = {
    #         "input_ids": inputs["input_ids_rejected"],
    #         "attention_mask": inputs["attention_mask_rejected"]
    #     }
    #     # Compute rewards via the implicit PRM model
    #     rewards_chosen = model(**chosen_inputs)["rewards"]
    #     rewards_rejected = model(**rejected_inputs)["rewards"]
    #     # Define loss as negative advantage
    #     loss = - (rewards_chosen - rewards_rejected).mean()
    #     if return_outputs:
    #         return loss, None
    #     return loss
    def compute_loss(self, model, inputs, return_outputs=False):
        # Prepare inputs
        chosen_inputs = {
            "input_ids": inputs["input_ids_chosen"],
            "attention_mask": inputs["attention_mask_chosen"]
        }
        rejected_inputs = {
            "input_ids": inputs["input_ids_rejected"],
            "attention_mask": inputs["attention_mask_rejected"]
        }

        # Forward pass
        outputs_chosen = model(**chosen_inputs)
        outputs_rejected = model(**rejected_inputs)

        # Extract rewards
        rewards_chosen = outputs_chosen["rewards"]
        rewards_rejected = outputs_rejected["rewards"]

        # Compute loss
        loss = - (rewards_chosen - rewards_rejected).mean()

        if return_outputs:
            # Return something like logits_dict
            return loss, {
                "rewards_chosen": rewards_chosen,
                "rewards_rejected": rewards_rejected
            }
        return loss

# Initialize the implicit PRM model
implicit_prm = ImplicitPRM(policy_model, ref_model, tokenizer, args.beta)

# PEFT配置 for policy model
if args.use_default_peft_config:
    peft_config_new = LoraConfig(
        r=4,
        lora_alpha=8,
        bias="none",
        task_type="SEQ_CLS",
        modules_to_save=["scores"],
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # 关键修正
    )
else:
    peft_config_new = None
    
# Apply PEFT if needed
if args.use_peft:
    implicit_prm = get_peft_model(implicit_prm, peft_config_new)
implicit_prm.print_trainable_parameters()

# Metrics setnt_trainable_parameters()

# Metrics setup
accuracy = load_metric('./accuracy.py', trust_remote_code=True)

def compute_metrics(eval_pred):
    predictions, _ = eval_pred
    predictions = np.argmax(predictions, axis=0)
    labels = np.zeros(predictions.shape)
    return accuracy.compute(predictions=predictions, references=labels)

# Initialize the custom trainer
trainer = PRMTrainer(
    model=implicit_prm,
    tokenizer=tokenizer,
    args=args.reward_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config_new,
    # compute_metrics=compute_metrics,
)

# Add callback for early evaluation
if args.eval_first_step:
    class EvaluateFirstStepCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 1:
                control.should_evaluate = True
    trainer.add_callback(EvaluateFirstStepCallback())

# Train the implicit PRM
trainer.train()
trainer.save_model()