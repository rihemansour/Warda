import numpy as np 
import json
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from nltk_utils import bag_of_words, tokenize, stem
from ChatDataset_class import ChatDataset

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
ignore_words = ['?', '.', '!',',',';',':','(',')','/','a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'have', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with','again','about','after']
all_words= [stem(word)for word in all_words if word not in ignore_words]
all_words= sorted(set(all_words))
tags=sorted(set(tags))
#print(len(xy), "patterns")
#print(len(tags), "tags:", tags)
#print(len(all_words), "unique stemmed words:", all_words)


# create training data

X_train = []
y_train = []

for (pattern_sentence,tag) in xy:
    bag = bag_of_words(pattern_sentence,all_words)
    print(bag)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)
    
X_train = np.array(X_train)
y_train = np.array(y_train)

   
batch_size = 8
dataset = ChatDataset(X_train=X_train,y_train=y_train)
train_loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=0)