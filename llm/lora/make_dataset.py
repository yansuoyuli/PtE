import os

os.environ["WANDB_DISABLED"] = "true"

from datasets import Dataset, load_dataset
import random
import pickle
import csv
import re

# ====================================================

# generate ground truth csv

# with open('~data/netflix/gpt_output_dict_item_side_filter.pkl', 'rb') as f:
#     gpt_output_dict_filter = pickle.load(f)

# data = {'UID': [],
#         'Input': [],
#         'Response': []}

# for uid in gpt_output_dict_filter:
#     data['UID'].append(uid)
#     data['Input'].append(gpt_output_dict_filter[uid]['input'])
#     data['Response'].append(gpt_output_dict_filter[uid]['output'])

# with open('./../../data/netflix/netflix_fine_tune.csv', 'w', newline='') as f:
#     writer = csv.DictWriter(f, fieldnames=data.keys())
#     writer.writeheader()
#     for i in range(len(data['UID'])):
#         row = {key: data[key][i] for key in data.keys()}
#         writer.writerow(row)

# ====================================================

# .csv -> hf data

dataset = load_dataset("csv", data_files="./../../data/netflix/item_side_instruction_data.csv", split="train")

print(f"dataset size: {len(dataset)}")
print(dataset[random.randrange(len(dataset))])

dataset.save_to_disk('./../../data/netflix/item_side_instruction_hf')

====================================================

# generate end2end data
with open('./../../data/netflix/self_instruction_dict.pkl', 'rb') as f:
    self_instruction_dict = pickle.load(f)

# with open('./../../data/mind/user_item_dict_9.pkl', 'rb') as f:
with open('./../../data/netflix/user_item_dict_train.pkl', 'rb') as f:
    user_item_dict_9 = pickle.load(f)

with open('./../../data/netflix/item_info_dict.pkl', 'rb') as f:
    item_info_dict = pickle.load(f)

# with open('./../../data/mind/user_profile/user_profile_dict_v0.pkl', 'rb') as f:
with open('./../../data/netflix/user_profile/llama_lora_netflix_v0/user_profile_dict.pkl', 'rb') as f:
    user_profile_dict = pickle.load(f)

data = {'UID': [],
        'Input1': [],
        'Response1': [],
        'Input2': [],
        'Response2': []}

counter = 0

for uid in self_instruction_dict:
    u_i_dict = {}
    current_user_item_list = user_item_dict_9[uid].copy()
    if len(current_user_item_list) > 5:  # only for netflix
        current_user_item_list = random.sample(current_user_item_list, 5)
    # u_i_dict[uid] = str(user_item_dict_9[uid])
    random.shuffle(current_user_item_list)
    # u_i_dict[uid] = str(current_user_item_list[1:])
    random_number = random.random()
    user_nb_list = self_instruction_dict[uid]
    item_list = []
    for uid_nb in user_nb_list:
        nb_item_list = user_item_dict_9[uid_nb]
        if len(nb_item_list) > 5: # only for netflix
            nb_item_list = random.sample(nb_item_list, 5)
        # u_i_dict[uid_nb] = str(user_item_dict_9[uid_nb])
        u_i_dict[uid_nb] = str(nb_item_list)
        for iid in nb_item_list:
            item_list.append(iid)
    item_list = list(set(item_list))
    if current_user_item_list[0] in item_list:
        counter += 1
    else:
        continue
    for iid in current_user_item_list[1:]:
        item_list.append(iid)
    item_list = list(set(item_list))
    input1_content = "Each user's historical item interaction list is as follows:\n"
    response1_content = ""
    input2_content = ""
    response2_content = ""
    input1_content += '[user ' + str(uid) + ', item interaction list:'
    input1_content += str(current_user_item_list[1:])
    for uid_ in u_i_dict:
        input1_content += ';\n user ' + str(uid_) + ', item interaction list:' + str(u_i_dict[uid_])
    # input1_content += '].\n The detail (category, subcategory, title, and abstract) of each item is as follows: [\n'
    input1_content += '].\n The detail (year and title) of each item is as follows: [\n'
    for iid in item_list:
        # input1_content += 'item ' + str(iid) + ', [category: ' + item_info_dict[iid]['cate'] + ', subcategory: ' + item_info_dict[iid]['sub_cate'] + ', title: ' + item_info_dict[iid]['title'] + ', abstract: ' + item_info_dict[iid]['abstract'].replace('...', '') + ']\n '
        input1_content += 'item ' + str(iid) + ', [year: ' + str(item_info_dict[iid]['year']) + ', title: ' + item_info_dict[iid]['title'] + ']\n '
    input1_content += ']. '
    input1_content += 'Please provide the profile strictly in the following format: User identity: [Identity 1], [Identity 2], [Identity 3]; User interests: [Interest 1], [Interest 2], [Interest 3].'
    response1_content += 'user ' + str(uid) + ', profile: [' + user_profile_dict[uid].strip() + ']\n'
    if len(user_profile_dict[uid].strip()) < 10:
        counter -= 1
        continue
    flag = True
    for uid_ in u_i_dict:
        response1_content += 'user ' + str(uid_) + ', profile: [' + user_profile_dict[uid_].strip() + ']\n'
        if len(user_profile_dict[uid_].strip()) < 10:
            flag = False
    if not flag:
        counter -= 1
        continue
    target_iid = None
    answer = None
    if random_number > 0.5:
        target_iid = current_user_item_list[0]
        answer = 'Yes'
    else:
        while True:
            target_iid = random.choice(item_list)
            if target_iid not in current_user_item_list:
                break
        answer = 'No'
    input2_content += f'Based on the user preferences and item information mentiond above, using collaborative filtering method, please determine whether user {str(uid)} will interact with item {str(target_iid)}. Just answer Yes or No.'
    response2_content += f'{answer}'
    
    data['UID'].append(uid)
    data['Input1'].append(input1_content)
    data['Response1'].append(response1_content)
    data['Input2'].append(input2_content)
    data['Response2'].append(response2_content)

print(counter)

with open('./../../data/netflix/cf_instruction_data.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=data.keys())
    writer.writeheader()
    for i in range(len(data['UID'])):
        row = {key: data[key][i] for key in data.keys()}
        writer.writerow(row)

# ============================================================================

# generate data for reward modeling

# with open('./../../data/netflix/self_instruction_dict.pkl', 'rb') as f:
#     self_instruction_dict = pickle.load(f)

# # with open('./../../data/mind/user_item_dict_9.pkl', 'rb') as f:
# with open('./../../data/netflix/user_item_dict_train.pkl', 'rb') as f:
#     user_item_dict_9 = pickle.load(f)

# with open('./../../data/netflix/item_info_dict.pkl', 'rb') as f:
#     item_info_dict = pickle.load(f)

# # with open('./../../data/mind/user_profile/user_profile_dict_v0.pkl', 'rb') as f:
# with open('./../../data/netflix/user_profile/llama_lora_netflix_v0/user_profile_dict.pkl', 'rb') as f:
#     user_profile_dict = pickle.load(f)

# data_train = {'UID': [],
#         'query': [],
#         'chosen': [],
#         'rejected': []}

# data_eval = {'UID': [],
#         'query': [],
#         'chosen': [],
#         'rejected': []}

# data_rl = {'UID': [],
#         'query': [],
#         'chosen': [],
#         'rejected': []}

# counter_train = 0
# counter_eval = 0
# counter_rl = 0

# counter = 0

# for uid in self_instruction_dict:
#     u_i_dict = {}
#     current_user_item_list = user_item_dict_9[uid].copy()
#     if len(current_user_item_list) > 5:  # only for netflix
#         current_user_item_list = random.sample(current_user_item_list, 5)
#     # u_i_dict[uid] = str(user_item_dict_9[uid])
#     random.shuffle(current_user_item_list)
#     # u_i_dict[uid] = str(current_user_item_list[1:])
#     random_number = random.random()
#     user_nb_list = self_instruction_dict[uid]
#     item_list = []
#     for uid_nb in user_nb_list:
#         nb_item_list = user_item_dict_9[uid_nb]
#         if len(nb_item_list) > 5: # only for netflix
#             nb_item_list = random.sample(nb_item_list, 5)
#         # u_i_dict[uid_nb] = str(user_item_dict_9[uid_nb])
#         u_i_dict[uid_nb] = str(nb_item_list)
#         for iid in nb_item_list:
#             item_list.append(iid)
#     # 
#     for iid in current_user_item_list:
#         item_list.append(iid)
#     # 
#     item_list = list(set(item_list))
#     # if current_user_item_list[0] in item_list:
#     #     counter += 1
#     # else:
#     #     continue
#     input1_content = "Each user's historical item interaction list is as follows:\n"
#     response1_content = ""
#     response2_content = ""
#     input1_content += '[Target user ' + str(uid) + ', item interaction list:'
#     input1_content += str(current_user_item_list)
#     for uid_ in u_i_dict:
#         input1_content += ';\n user ' + str(uid_) + ', item interaction list:' + str(u_i_dict[uid_])
#     # input1_content += '].\n The detail (category, subcategory, title, and abstract) of each item is as follows: [\n'
#     input1_content += '].\n The detail (year and title) of each item is as follows: [\n'
#     # input1_content += '].\n The detail (category, subcategory, and title) of each item is as follows: [\n'
#     for iid in item_list:
#         # input1_content += 'item ' + str(iid) + ', [category: ' + item_info_dict[iid]['cate'] + ', subcategory: ' + item_info_dict[iid]['sub_cate'] + ', title: ' + item_info_dict[iid]['title'] + ', abstract: ' + item_info_dict[iid]['abstract'].replace('...', '') + ']\n '
#         input1_content += 'item ' + str(iid) + ', [year: ' + str(item_info_dict[iid]['year']) + ', title: ' + item_info_dict[iid]['title'] + ']\n '
#         # input1_content += 'item ' + str(iid) + ', [category: ' + item_info_dict[iid]['cate'] + ', subcategory: ' + item_info_dict[iid]['sub_cate'] + ', title: ' + item_info_dict[iid]['title'] + ']\n '
#     input1_content += ']. '
#     # input1_content += f'Please provide the target user {str(uid)}\'s profile strictly in the following format: User identity: [Identity 1], [Identity 2], [Identity 3]; User interests: [Interest 1], [Interest 2], [Interest 3].'
#     input1_content += f'Please provide the target user {str(uid)}\'s profile.'
#     response1_content += user_profile_dict[uid].strip()
#     if len(user_profile_dict[uid].strip()) < 10:
#         continue
#     random_number_neg = random.random()
#     false_user_profile = random.choice(user_nb_list)
#     if random_number_neg < 0.3:
#         response2_content += user_profile_dict[false_user_profile].strip()  
#     elif random_number_neg >= 0.3 and random_number_neg < 0.6:
#         response2_content += user_profile_dict[uid].strip()
#         response2_content += '\n'
#         response2_content += user_profile_dict[false_user_profile].strip()
#     elif random_number_neg >= 0.6 and random_number_neg < 0.7:
#         postive_string = user_profile_dict[uid].strip()
#         identity_pattern = "User identity:\s*(.*?);"
#         interests_pattern = "User interests:\s*(.*?);"
#         identity_matches = re.findall(identity_pattern, postive_string)
#         interests_matches = re.findall(interests_pattern, postive_string)
#         if identity_matches:
#             neg_string = "User identity: " + identity_matches[0]
#         else:
#             neg_string = "User identity: "
#         response2_content += neg_string
#     elif random_number_neg >= 0.7 and random_number_neg < 0.8:
#         postive_string = user_profile_dict[uid].strip()
#         identity_pattern = "User identity:\s*(.*?);"
#         interests_pattern = "User interests:\s*(.*?);"
#         identity_matches = re.findall(identity_pattern, postive_string)
#         interests_matches = re.findall(interests_pattern, postive_string)
#         if identity_matches:
#             neg_string = "User identity: " + identity_matches[0] + identity_matches[0]
#         else:
#             neg_string = "User interest: "
#         response2_content += neg_string
#     elif random_number_neg >= 0.8 and random_number_neg < 0.9:
#         postive_string = user_profile_dict[uid].strip()
#         identity_pattern = "User identity:\s*(.*?);"
#         interests_pattern = "User interests:\s*(.*?);"
#         identity_matches = re.findall(identity_pattern, postive_string)
#         interests_matches = re.findall(interests_pattern, postive_string)
#         neg_string = "User identity: "
#         response2_content += neg_string
#     else:
#         response2_content += '.'
#     if random_number < 0.2:
#         data_eval['UID'].append(uid)
#         data_eval['query'].append(input1_content)
#         data_eval['chosen'].append(response1_content)
#         data_eval['rejected'].append(response2_content)
#         counter_eval += 1
#     elif random_number >= 0.2 and random_number < 0.6:
#         data_train['UID'].append(uid)
#         data_train['query'].append(input1_content)
#         data_train['chosen'].append(response1_content)
#         data_train['rejected'].append(response2_content)
#         counter_train += 1
#     else:
#         data_rl['UID'].append(uid)
#         data_rl['query'].append(input1_content)
#         data_rl['chosen'].append(response1_content)
#         data_rl['rejected'].append(response2_content)
#         counter_rl += 1

# print(counter_train)
# print(counter_eval)
# print(counter_rl)

# with open('./../../data/netflix/rlhf/netflix_data_v0/train.csv', 'w', newline='') as f:
#     writer = csv.DictWriter(f, fieldnames=data_train.keys())
#     writer.writeheader()
#     for i in range(len(data_train['UID'])):
#         row = {key: data_train[key][i] for key in data_train.keys()}
#         writer.writerow(row)

# with open('./../../data/netflix/rlhf/netflix_data_v0/eval.csv', 'w', newline='') as f:
#     writer = csv.DictWriter(f, fieldnames=data_eval.keys())
#     writer.writeheader()
#     for i in range(len(data_eval['UID'])):
#         row = {key: data_eval[key][i] for key in data_eval.keys()}
#         writer.writerow(row)

# with open('./../../data/netflix/rlhf/netflix_data_v0/rl.csv', 'w', newline='') as f:
#     writer = csv.DictWriter(f, fieldnames=data_rl.keys())
#     writer.writeheader()
#     for i in range(len(data_rl['UID'])):
#         row = {key: data_rl[key][i] for key in data_rl.keys()}
#         writer.writerow(row)

# ============================================================================

# generate data for item side profile generator

# # with open('./../../data/mind/item_info_dict.pkl', 'rb') as f:
# # with open('./../../data/netflix/item_info_dict.pkl', 'rb') as f:
#     item_info_dict = pickle.load(f)

# # with open('./../../data/mind/self_instruction_dict_item.pkl', 'rb') as f:
# # with open('./../../data/netflix/self_instruction_dict_item.pkl', 'rb') as f:  
#     self_instruction_dict_item = pickle.load(f)

# # with open('./../../data/mind/gpt_output_dict_item_side_filter.pkl', 'rb') as f:
# # with open('./../../data/netflix/gpt_output_dict_item_side_filter.pkl', 'rb') as f:
#     item_profile_dict = pickle.load(f)

# data = {
#     'UID': [],
#     'Input': [],
#     'Response': []
# }


# print(len(self_instruction_dict_item))
# print(len(item_profile_dict))

# for iid in self_instruction_dict_item:
#     content = "The target item text information: "
#     # mind
#     # content += str(item_info_dict[iid])[1:-1]
#     # content += '[cate: '
#     # content += item_info_dict[iid]['cate']
#     # content += ', sub_cate: '
#     # content += item_info_dict[iid]['sub_cate']
#     # content += ', title: '
#     # content += item_info_dict[iid]['title']
#     # content += ', abstract: '
#     # content += item_info_dict[iid]['abstract']
#     # netflix
#     # content += '[title: '
#     # content += item_info_dict[iid]['title']
#     # content += ', year: '
#     # content += str(item_info_dict[iid]['year'])
#     content += '[title: '
#     content += item_info_dict[iid]
#     content += ']. Here is the list of item (text information and the profile of the users that the item is targeting):\n'
#     for iid_ in self_instruction_dict_item[iid]:
#         content += 'item '
#         content += str(iid_)
#         content += ': ['
#         content += 'Text information: '
#         # mind
#         # content += str(item_info_dict[iid_])[1:-1]
#         # content += 'cate: '
#         # content += item_info_dict[iid_]['cate']
#         # content += ', sub_cate: '
#         # content += item_info_dict[iid_]['sub_cate']
#         # content += ', title: '
#         # content += item_info_dict[iid_]['title']
#         # netflix
#         # content += 'title: '
#         # content += item_info_dict[iid_]['title']
#         # content += ', year: '
#         # content += str(item_info_dict[iid_]['year'])
#         content += 'title: '
#         content += item_info_dict[iid_]
#         content += ', the profile of the users that the item is targeting: '
#         content += item_profile_dict[iid_]['output']
#         content += '];\n'
    
#     data['UID'].append(iid)
#     data['Input'].append(content)
#     data['Response'].append(item_profile_dict[iid]['output'])

# with open('./../../data/netflix/item_side_instruction_data.csv', 'w', newline='') as f:
#     writer = csv.DictWriter(f, fieldnames=data.keys())
#     writer.writeheader()
#     for i in range(len(data['UID'])):
#         row = {key: data[key][i] for key in data.keys()}
#         writer.writerow(row)