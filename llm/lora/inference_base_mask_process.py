import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["WANDB_DISABLED"] = "true"

import torch
import pickle
from random import randrange
from datasets import load_from_disk
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
import random

class ChatBot:
    def __init__(self, model_path):
        self.model = LlamaForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            load_in_4bit=True,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.tokenizer.clean_up_tokenization_spaces = True


    def get_response(self, message):
        input_ids = self.tokenizer(message, return_tensors="pt", truncation=True).input_ids.cuda()
        outputs = self.model.generate(input_ids=input_ids, max_new_tokens=100, do_sample=True, top_p=0.9,temperature=0.9, repetition_penalty=1.3)
        return self.tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(message):]

    def chat_test(self, str_):
        response = self.get_response(str_)
        print(response)

    def chat(self, user_item_dict, item_info_dict, uid, uid_nb_list, result_dict):
        content_begin = "[INST] <<SYS>>\n You are a recommendation system capable of predicting user-item interactions based on the principles of collaborative filtering. Specifically, it can be devided into two stages. In the first stage, you will generate a user preference profile based on the user's historical behavior. In the second stage, using the preference profile generated in the first stage, you will find users with similar preferences and apply their historical interaction records to the target user. This allows you to determine whether the target user is likely to interact with a particular item in the future. \n<</SYS>>\n\n"
        content_end = f"Please provide the profile of the target user {str(uid)} strictly in the following format: User identity: [Identity 1], [Identity 2], [Identity 3]; User interests: [Interest 1], [Interest 2], [Interest 3]. Emphasize that only the most likely three identities and interests of the target user should be provided, and strictly adhere to the above format.[/INST]"

        content_item_list = "Each user's historical item interaction list is as follows:\n"
        content_item_list += '[Target user ' + str(uid) + '\'s interaction list:'
        target_user_item_list = user_item_dict[uid]
        # if len(target_user_item_list) > 5: # only for netflix
        #     target_user_item_list = random.sample(target_user_item_list, 5)
        # content_item_list += str(user_item_dict[uid])
        content_item_list += str(target_user_item_list)
        temp_item_list = list()
        # for iid in user_item_dict[uid]:
        #     temp_item_list.append(iid)
        for iid in target_user_item_list:
            temp_item_list.append(iid)
        for uid_ in uid_nb_list:
            nb_item_list = user_item_dict[uid_]
            # if len(nb_item_list) > 5: # only for netflix
            #     nb_item_list = random.sample(nb_item_list, 5)
            # for iid in user_item_dict[uid_]:
            for iid in nb_item_list:
                temp_item_list.append(iid)
            # content_item_list += ';\n user ' + str(uid_) + ', item interaction list:' + str(user_item_dict[uid_])
            content_item_list += ';\n user ' + str(uid_) + ', item interaction list:' + str(nb_item_list)
        content_item_list += '.\n'
        content_item_list += 'The detail (category, subcategory, title, and abstract) of each item is as follows: [\n'
        # content_item_list += 'The detail (year and title) of each item is as follows: [\n'
        temp_item_list = list(set(temp_item_list))
        for iid in temp_item_list:
            #content_item_list += 'item ' + str(iid) + ', [category: ' + item_info_dict[iid]['cate'] + ', subcategory: ' + item_info_dict[iid]['sub_cate'] + ', title: ' + item_info_dict[iid]['title'] + ', abstract: ' + item_info_dict[iid]['abstract'].replace('...', '') + ']\n '
            content_item_list += 'item ' + str(iid) + ', [year: ' + str(item_info_dict[iid]['year']) + ', title: ' + item_info_dict[iid]['title'] + ']\n '
        content_item_list += ']. '
        content = content_begin + content_item_list + content_end

        # print("SEND: ")
        # print(content)

        response = self.get_response(content)
        result_dict[uid] = response

        # print("GPT: ")
        # print(response)
        # print()


file_user_item_dict = '~data/netflix/user_item_dict_train.pkl'
# file_user_item_dict = './../../data/netflix/user_item_dict_train.pkl'
with open(file_user_item_dict, 'rb') as fs:
    user_item_dict = pickle.load(fs)

file_item_info_dict = '~data/netflix/item_info_dict.pkl'
# file_item_info_dict = './../../data/netflix/item_info_dict.pkl'
with open(file_item_info_dict, 'rb') as fs:
    item_info_dict = pickle.load(fs)

with open('~data/netflix/self_instruction_dict.pkl', 'rb') as f:
# with open('./../../data/netflix/self_instruction_dict.pkl', 'rb') as f:
    self_instruction_dict = pickle.load(f)

result_dict = {}
chatbot = ChatBot("~llm/ft_models/rlhf/netflix_final_rlhf_testmaskv1_output_process_batch_4_fixed_step_2000")
# chatbot = ChatBot("./../ft_models/netflix/user/merged_model_5385")

counter = 0

print(len(user_item_dict))
from tqdm import tqdm
for uid in tqdm(self_instruction_dict):
    uid_nb_list = self_instruction_dict[uid]
    chatbot.chat(user_item_dict, item_info_dict, uid, uid_nb_list, result_dict)
    #print(counter)
    counter += 1
    torch.cuda.empty_cache()

print(counter)

output_f = open('~data/netflix/user_profile/netflix_final_rlhf_testmaskv1_batch_4_step_2000/user_profile_dict.pkl', 'wb')
pickle.dump(result_dict, output_f)
output_f.close()
