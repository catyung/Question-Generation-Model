import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AdamW,
    MT5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
from utils import T5PegasusTokenizer
from transformers.models.mt5.modeling_mt5 import MT5ForConditionalGeneration
from datasets import load_dataset

# GPU setting , if you have no GPU, please use 'cpu'
device = torch.device('cuda')

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
set_seed(42)

# Loading Model
model_path = 'imxly/t5-pegasus'
mt5_model = MT5ForConditionalGeneration.from_pretrained(model_path).to(device)
tokenizer = T5PegasusTokenizer.from_pretrained(model_path)

# Loading Dataset from Huggingface 
data = load_dataset("xquad","xquad.zh")
data = data['validation']

data = pd.DataFrame(data)

data['question'] = data['question']
data['context'] = data['context']
data['answers'] = list(map(lambda x: x['text'][0], data['answers']))

data['input'] = 'question: '+'<answer>' + data['answers'] + '<context>' + data['context']
data['label'] = data['question']

input_data = list(zip(data['input'],data['label']))

# optimizer
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in t5_model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
    {
        "params": [p for n, p in t5_model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]

optimizer = AdamW(optimizer_grouped_parameters, lr=3e-4, eps=1e-8)

# Training (This is the basic setting of training, recommended to use TrainingArgs to setup)
mt5_model.train()

epochs = 5

for epoch in range(epochs):
  print ("epoch ",epoch)
  for input,output in input_data:
    # Setting prefix for task 
    input_sent = "qa-generation :"+input+ "</s> "
    ouput_sent = output+"</s>  "

    tokenized_inp = tokenizer.encode_plus(input_sent,  max_length=512, pad_to_max_length=True,return_tensors="pt")
    tokenized_output = tokenizer.encode_plus(ouput_sent, max_length=100, pad_to_max_length=True,return_tensors="pt")

    input_ids  = tokenized_inp["input_ids"].to(device)
    attention_mask = tokenized_inp["attention_mask"].to(device)

    lm_labels= tokenized_output["input_ids"].to(device)
    decoder_attention_mask=  tokenized_output["attention_mask"].to(device)

    # the forward function automatically creates the correct decoder_input_ids
    output = mt5_model(input_ids=input_ids, labels=lm_labels,decoder_attention_mask=decoder_attention_mask,attention_mask=attention_mask)
    loss = output[0]

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Saving the model & tokenizer
mt5_model.save_pretrained("final-mt5")
tokenizer.save_pretrained("final-mt5")
