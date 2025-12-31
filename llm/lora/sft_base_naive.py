import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["WANDB_DISABLED"] = "true"

import torch
from random import randrange
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import TrainingArguments
from trl import SFTTrainer


def format_instruction(sample):
	return f"""
    <s>[INST] <<SYS>>
	You are a recommendation system capable of predicting user-item interactions based on the principles of collaborative filtering. Specifically, it can be devided into two stages.
	In the first stage, you will generate a user preference profile based on the user's historical behavior.
	In the second stage, using the preference profile generated in the first stage, you will find users with similar preferences and apply their historical interaction records to the target user.
	This allows you to determine whether the target user is likely to interact with a particular item in the future.
    <</SYS>>
    {sample['Input1']}[/INST]
    {sample['Response1']}</s>
    <s>[INST]{sample['Input2']}[/INST]
    {sample['Response2']}</s>
    """

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

dataset = load_from_disk("./../../sft_data/mind/cf_instruction_hf")

print(format_instruction(dataset[randrange(len(dataset))]))

use_flash_attention = False
model_id = "./../ft_models/llama_lora_mind_v0/merged_model"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    use_cache=False,
    use_flash_attention_2=use_flash_attention,
    device_map='auto',
)
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(model_id)
# tokenizer.pad_token = tokenizer.eos_token
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
    output_dir="./../ft_models/llama_lora_mind_navie_v1",
    num_train_epochs=3,
    # num_train_epochs=1,
    per_device_train_batch_size=6 if use_flash_attention else 1,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-4,
    bf16=True,
    tf32=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    disable_tqdm=False # disable tqdm since with packing values are in correct
)

max_seq_length = 4096 # max sequence length for model and packing of the dataset

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    formatting_func=format_instruction,
    args=args,
)

trainer.train()
trainer.save_model()
