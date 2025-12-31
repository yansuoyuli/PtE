import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["WANDB_DISABLED"] = "true"

import torch
from random import randrange
from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import TrainingArguments
from trl import SFTTrainer

from transformers.trainer_pt_utils import LabelSmoother
from torch.utils.data import Dataset
import transformers
import copy
from typing import Dict, Optional, Sequence
from dataclasses import dataclass, field

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

local_rank = None

TRAINING_ARGS_NAME = "training_args.bin"
MAX_SEQ_LENGTH = 2048

class LoRATrainer(transformers.Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving model checkpoint to {output_dir}")

        self.model.save_pretrained(
            output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
        )
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

def rank0_print(*args):
     if local_rank == 0:
          print(*args)

def last_index(lst, value):
    return next((len(lst) - i - 1 for i, x in enumerate(lst[::-1]) if x != value), -1)

def safe_ids(ids, max_value, pad_id):
     return [i if i < max_value else pad_id for i in ids]

def mask_tokenize(item, tokenizer):
    input_ids = []
    labels = []
    system = 'You are a recommendation system capable of predicting user-item interactions based on the principles of collaborative filtering. Specifically, it can be devided into two stages. In the first stage, you will generate a user preference profile based on the user\'s historical behavior. In the second stage, using the preference profile generated in the first stage, you will find users with similar preferences and apply their historical interaction records to the target user. This allows you to determine whether the target user is likely to interact with a particular item in the future.'
    system = B_SYS + system + E_SYS

    content_input1 = system + item["Input1"] 
    content_response1 = item['Response1']
    content_input2 = item['Input2']
    content_response2 = item['Response2']

    content_input1 = f"{B_INST} {content_input1} {E_INST}"
    content_input1_ids = tokenizer.encode(content_input1)
    labels += [LabelSmoother.ignore_index] * (len(content_input1_ids))
    input_ids += content_input1_ids

    content_response1 = f"{content_response1}"
    content_response1_ids = tokenizer.encode(content_response1, add_special_tokens=False) + [tokenizer.eos_token_id]
    labels += content_response1_ids
    input_ids += content_response1_ids

    content_input2 = f"{B_INST} {content_input2} {E_INST}"
    content_input2_ids = tokenizer.encode(content_input2)
    labels += [LabelSmoother.ignore_index] * (len(content_input2_ids))
    input_ids += content_input2_ids

    content_response2 = f"{content_response2}"
    content_response2_ids = tokenizer.encode(content_response2, add_special_tokens=False) + [tokenizer.eos_token_id]
    labels += content_response2_ids
    input_ids += content_response2_ids

    input_ids = input_ids[:tokenizer.model_max_length]
    labels = labels[:tokenizer.model_max_length]

    trunc_id = last_index(labels, LabelSmoother.ignore_index) + 1
    input_ids = input_ids[:trunc_id]
    labels = labels[:trunc_id]
    input_ids = safe_ids(input_ids, tokenizer.vocab_size, tokenizer.pad_token_id)
    labels = safe_ids(labels, tokenizer.vocab_size, LabelSmoother.ignore_index)
    return input_ids, labels

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        item = self.raw_data[i]
        input_ids, labels = mask_tokenize(
            copy.deepcopy(item),
            self.tokenizer)
        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)
        ret = dict(
            input_ids=input_ids,
            labels=labels,
        )
        self.cached_data_dict[i] = ret

        return ret
    
@dataclass
class DataCollatorForSupervisedDataset(object):
     tokenizer: transformers.PreTrainedTokenizer
     def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
          )
        labels = torch.nn.utils.rnn.pad_sequence(
             labels,
             batch_first=True,
             padding_value=LabelSmoother.ignore_index
        )
        return dict(
             input_ids=input_ids,
             labels=labels,
             attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer):
    rank0_print("Loading data...")
    
    # ft_dataset = load_dataset("csv", data_files="./../../sft_data/mind/cf_instruction_data.csv", split="train")
    ft_dataset = load_dataset("csv", data_files="./../../sft_data/netflix/cf_instruction_data.csv", split="train")
    train_dataset = LazySupervisedDataset(ft_dataset, tokenizer=tokenizer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return train_dataset, data_collator

use_flash_attention = False
# model_id = "./../ft_models/mind/llama_lora_user_base/merged_model_105"
model_id = "~llm/ft_models/netflix/llama_lora_user_base/checkpoint-75"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    use_cache=False,
    use_flash_attention_2=use_flash_attention,
    # device_map='auto',
)
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side = "right"

peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=8,
        bias="none",
        task_type="CAUSAL_LM",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

args = TrainingArguments(
    # output_dir="./../ft_models/mind/llama_lora_user_mask",
    output_dir="./../ft_models/netflix/llama_lora_user_mask",
    num_train_epochs=3,
    # num_train_epochs=1,
    # per_device_train_batch_size=6 if use_flash_attention else 4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    # gradient_checkpointing_kwargs={"use_reentrant": False},
    # optim="paged_adamw_32bit",
    optim="adamw_torch",
    logging_steps=100,
    save_strategy="epoch",
    learning_rate=5e-5,
    bf16=True,
    tf32=True,
    # max_grad_norm=0.3,
    # warmup_ratio=0.03,
    # lr_scheduler_type="constant",
    disable_tqdm=False # disable tqdm since with packing values are in correct
)

# max_seq_length = 2048 # max sequence length for model and packing of the dataset

train_dataset, data_collator = make_supervised_data_module(tokenizer)


trainer = LoRATrainer(
    model=model,
    train_dataset=train_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    args=args,
)

trainer.train()
trainer.save_model()
