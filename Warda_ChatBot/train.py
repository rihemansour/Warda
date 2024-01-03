import numpy as np 
import json
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem


with open('intents.json', 'r') as f:
    intents = json.load(f)
    
all_words = []
tags = []
xy = []

# loop through each sentence in our intents patterns

for intent in intents['intents']:
    tag = intent["tags"]
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w,tag))
        
# stem and lower each word
ignore_words = ['?', '.', '!',',',';',':','(',')','a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'have', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with','again','about','after']
all_words= [stem(word)for word in all_words if word not in ignore_words]
all_words= sorted(set(all_words))
tags=sorted(set(tags))
print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)
