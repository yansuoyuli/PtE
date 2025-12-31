from peft import AutoPeftModelForCausalLM, PeftModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# model = AutoPeftModelForCausalLM.from_pretrained(
#     "./../ft_models/netflix/llama_lora_item/checkpoint-975",
#     low_cpu_mem_usage=True,
# )
base_model = AutoModelForSequenceClassification.from_pretrained("~/models/Llama-2-7b-chat-hf")
model = PeftModel.from_pretrained(base_model, "~llm/ft_models/netflix/llama_lora_item/checkpoint-975")


tokenizer = AutoTokenizer.from_pretrained("~/models/Llama-2-7b-chat-hf")

# Merge LoRA and base model
merged_model = model.merge_and_unload()

# Save the merged model
merged_model.save_pretrained("~llm/ft_models/netflix/llama_lora_item_merge/",safe_serialization=True)
tokenizer.save_pretrained("~llm/ft_models/netflix/llama_lora_item_merge/")
