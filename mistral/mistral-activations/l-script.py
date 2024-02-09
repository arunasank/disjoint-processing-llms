from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig,
    BitsAndBytesConfig
)
import bitsandbytes
from accelerate import infer_auto_device_map
from tqdm import tqdm
import sys
from datasets import Dataset, load_dataset
import torch
import random
import pandas as pd
COL_NUM = int(sys.argv[1])
#LAYER = int(sys.argv[2])
MODEL_PATH = '/home/gridsan/arunas/models/mistralai/Mistral-7B-v0.1/'
TOKENIZER_PATH = '/home/gridsan/arunas/tokenizers/mistralai/Mistral-7B-v0.1/'

model_path = f"{MODEL_PATH}"
tokenizer_path = f'{TOKENIZER_PATH}'
device="cuda:0"

config = AutoConfig.from_pretrained(model_path)

tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_path, config=config, device_map="auto", padding_side="left"
)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(f'{model_path}', device_map="auto", load_in_4bit=True)

device_map = infer_auto_device_map(model)

def get_activation(sentence, layer_num):
    tokens = tokenizer(sentence, return_tensors='pt')
    outputs = model(**tokens, output_hidden_states=True, output_attentions=True)
    layer_activations = outputs['hidden_states'][layer_num]
    return layer_activations

ngs = pd.read_csv('/home/gridsan/arunas/broca/ngs.csv')

ngs.drop(labels=['Unnamed: 0'], inplace=True, axis='columns')
ngs = ngs[['sentence', 'subordinate-sentence', 'passive-sentence', 'it','it-r-1-null_subject', 'it-r-2-passive', 'it-r-3-subordinate','it-u-1-negation', 'it-u-2-invert', 'it-u-3-gender', 'jp-r-1-sov','jp-r-2-passive', 'jp-r-3-subordinate', 'jp-u-1-negation','jp-u-2-invert', 'jp-u-3-past-tense']]
col = ngs.columns[COL_NUM]
#layer = LAYER
mean_activations = {}
for layer in range(33):
    print(f'Layer {layer}')
    mean_activations[layer] = {}
    activations = {}
    for row in list(ngs[col][609:]):
        curr_activation = get_activation(row, layer)
        for token_pos in range(len(row.split(" "))):
            if (not token_pos in activations):
                activations[token_pos] = torch.unsqueeze(curr_activation[:, token_pos, :].clone(), dim=0)
            else:
                activations[token_pos] = torch.cat([activations[token_pos], torch.unsqueeze(curr_activation[:, token_pos, :].clone(), dim=0)], dim=0)
    for token_pos in range(len(row.split(" "))):
        mean_activations[layer][token_pos] = round(torch.mean(torch.mean(activations[token_pos], axis=0), axis=1).item(), 2)

print(col, mean_activations)
