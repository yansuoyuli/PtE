import os

os.environ["WANDB_DISABLED"] = "true"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import torch
import pickle
from random import randrange
from datasets import load_from_disk
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer

class ChatBot:
    def __init__(self, model_path):
        self.model = LlamaForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            load_in_4bit=True,
        )
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)

    def get_response(self, message):
        input_ids = self.tokenizer(message, return_tensors="pt", truncation=True).input_ids.cuda()
        outputs = self.model.generate(input_ids=input_ids, max_new_tokens=100, do_sample=True, top_p=0.9,temperature=0.1)
        return self.tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(message):]

    def chat(self, item_list, item_info_dict, uid, result_dict):
        # content_begin = "Now you are a user profile generator. I will provide you with a list of news articles that a user has clicked on in the past. Each news article contains four pieces of information: category, subcategory, title, and abstract. Based on this information, please generate the user's profile. Here is the list of previously clicked news articles: "
        content_begin = "Now you are a user profile generator. I will provide you with a list of movies that a user has watched in the past. Each movies contains two pieces of information: year and title. Based on this information, please generate the user's profile. Here is the list of previously watched movies: "
        # content_begin = "Now you are a user profile generator. I will provide you with a list of articles that a user has read in the past. Each articles contains their titles. Based on this information, please generate the user's profile. Here is the list of previously read articles: "
        content_end = "Please provide the profile strictly in the following format: User identity: [Identity 1], [Identity 2], [Identity 3]; User interests: [Interest 1], [Interest 2], [Interest 3]. Emphasize that only the most likely three identities and interests should be provided, and strictly adhere to the above format."

        content_item_list = ""
        counter_item = 0 # only for netflix dataset
        for iid in item_list:
            content_item_list += "["
            content_item_list += str(item_info_dict[iid])[1:-1]
            content_item_list += "] "
            counter_item += 1
            if counter_item >= 5:
                break
        content = content_begin + content_item_list + content_end

        print("SEND: ")
        print(content)

        response = self.get_response(content)
        result_dict[uid] = response

        print("GPT: ")
        print(response)
        print()


# file_user_item_dict = './../../data/mind/user_item_dict_train.pkl'
file_user_item_dict = '~data/netflix/user_item_dict_train.pkl'
with open(file_user_item_dict, 'rb') as fs:
    user_item_dict = pickle.load(fs)

# file_item_info_dict = './../../data/mind/item_info_dict.pkl'
file_item_info_dict = '~data/netflix/item_info_dict.pkl'
with open(file_item_info_dict, 'rb') as fs:
    item_info_dict = pickle.load(fs)

result_dict = {}
# chatbot = ChatBot("./../ft_models/mind/user/merged_model_105")
chatbot = ChatBot("~llm/ft_models/netflix/llama_lora_user_base/checkpoint-51")

counter = 0

print(len(user_item_dict))

for uid in user_item_dict:
    item_list = user_item_dict[uid]
    chatbot.chat(item_list, item_info_dict, uid, result_dict)
    print(counter)
    counter += 1

print(counter)

# output_f = open('./../../data/netflix/user_profile/llama_hf/user_profile_dict.pkl', 'wb')
# pickle.dump(result_dict, output_f)
# output_f.close()





