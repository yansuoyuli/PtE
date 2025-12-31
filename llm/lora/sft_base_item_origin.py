import os

os.environ["WANDB_DISABLED"] = "true"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import torch
from random import randrange
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import TrainingArguments
from trl import SFTTrainer

def format_instruction(sample):
	return f"""
	<s>[INST]
    Now you are a profile generator. I will provide you with textual information about one target item as well as a list of other items with their textual information and their targeting users' profiles. Based on this information, please generate the user profile of this target item.
    {sample['Input']}
    Please provide the profile with 5 identities and 5 interests strictly in the following format: User identity: [Identity 1], [Identity 2], [Identity 3], [Identity 4], [Identity 5]; User interests: [Interest 1], [Interest 2], [Interest 3], [Interest 4], [Interest 5].[/INST]
    {sample['Response']}</s>
    """

dataset = load_from_disk("~sft_data/netflix/item_side_instruction_hf")

print(format_instruction(dataset[randrange(len(dataset))]))

use_flash_attention = False
model_id = "~/models/Llama-2-7b-chat-hf"

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
    output_dir="~llm/ft_models/netflix/llama_lora_item_process_8",
    num_train_epochs=5,
    per_device_train_batch_size=6 if use_flash_attention else 1,
    gradient_accumulation_steps=1,
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
