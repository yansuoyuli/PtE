from sentence_transformers import SentenceTransformer
import pickle
import numpy as np
import re

model = SentenceTransformer('bert-base-nli-mean-tokens')

with open('./path/to/file/dict.pkl', 'rb') as f:
    info_dict = pickle.load(f)

embedding_dict = {}

counter = 0
for iid in info_dict:
    # embedding_dict[iid] = model.encode(info_dict[iid])
    embedding_dict[iid] = model.encode(info_dict[iid].strip())
    # embedding_dict[iid] = model.encode(str(info_dict[iid]).replace('{','').replace('}',''))

keys = np.array(list(embedding_dict.keys()))
values = np.array(list(embedding_dict.values()))

indexed_array = np.zeros((np.max(keys) + 1, 768))
indexed_array[keys] = values

np.save('./path/to/file/embedding.npy', indexed_array)