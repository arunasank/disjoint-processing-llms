from nnsight import LanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
import bitsandbytes
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import pickle
import sys
import os
import matplotlib.pyplot as plt
import traceback
import argparse
import yaml
import json
parser = argparse.ArgumentParser()

parser.add_argument('--config_file', type=str, help='path to the model training config file, found in broca/config')
parser.add_argument('--stype', type=int, help='structure type idx. Can range from 0-30')

args = parser.parse_args()
with open(args.config_file, 'r') as f:
    config_file = yaml.safe_load(f)

# print(json.dumps(config_file, indent=4))
PREFIX = config_file["prefix"]
MODEL_NAME = config_file["model_name"]
MODEL_PATH = config_file["model_path"]
DATA_PATH = config_file["data_path"]
PROMPT_FILES_PATH = config_file["prompt_files_path"]
PATCH_PICKLES_PATH = config_file["patch_pickles_path"]
PATCH_PICKLES_SUBPATH = config_file["patch_pickles_sub_path"]
DATATYPE = config_file["datatype"]
os.makedirs(f"{PATCH_PICKLES_PATH}/attn/{PATCH_PICKLES_SUBPATH}/", exist_ok=True)
os.makedirs(f"{PATCH_PICKLES_PATH}/mlp/{PATCH_PICKLES_SUBPATH}/", exist_ok=True)

og = pd.read_csv(DATA_PATH)
types = []
# types = [col for col in sorted(og.columns) if (('en' in col[:2]) or ('ita' in col[:3]) or ('jap' in col[:3])) and (not 'qsub' in col) and (not 'null_subject' in col)]
if DATATYPE == "nonce":
    types = sorted([col for col in sorted(og.columns) if not ('ng-' in col) and 'en_S-' in col])
elif DATATYPE == "conventional":
    types = sorted([col for col in og.columns \
            if not '_S' in col \
            and (not 'qsub' in col) and (not 'null_subject' in col) \
            and ('ita-' in col  or 'en-' in col or 'jap-' in col)])
elif DATATYPE == "conv-nonce":
    NONCE_DATA_PATH = config_file['nonce_data_path']
    NONCE_PROMPTS_PATH = config_file['nonce_prompts_path']
    nonce_data = pd.read_csv(NONCE_DATA_PATH)
    types = sorted([col for col in og.columns \
            if not '_S' in col \
            and (not 'qsub' in col) and (not 'null_subject' in col) \
            and ('ita-' in col  or 'en-' in col or 'jap-' in col)])
    nonceType = sorted([col for col in sorted(nonce_data.columns) \
            if not ('ng-' in col) \
            and 'en_S-' in col])[args.stype]
    
    
sType = col = types[args.stype]

if '7b' in MODEL_NAME or '8b' in MODEL_NAME or '70b' in MODEL_NAME:
    # os.environ["HF_TOKEN"] = config_file["hf_token"]
    # MODEL_CACHE_PATH = config_file["model_cache_path"]
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    config = AutoConfig.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, config=config, device_map="auto", padding_side="left")
    
    tokenizer.pad_token = tokenizer.eos_token
    model = LanguageModel(MODEL_PATH, quantization_config=nf4_config, tokenizer=tokenizer, device_map='auto') # Load the model

else:
    config = AutoConfig.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, config=config, device_map="auto", padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = LanguageModel(MODEL_PATH, tokenizer=tokenizer, device_map='auto') # Load the model
    
def get_prompt_from_df(filename):
    data = list(pd.read_csv(filename)['prompt'])
    questions = list(pd.read_csv(filename)['q'])
    golds = list(pd.read_csv(filename)['gold'])
    return data, questions, golds

def getPaddedTrainTokens(prompts, golds):
    global DATATYPE, n_prompts
    max_len = 0
    train_prefixes = []
    train_tokens = []
    
    for idx, prompt in enumerate(prompts):
        train_example = prompt
        train_prefixes.append(train_example)
        if DATATYPE == 'conv-nonce':
            # print('conv-nonce')
            nonce_example = f"{n_prompts[idx]}"
            max_len = max(max_len, len(model.tokenizer(nonce_example)['input_ids']), len(model.tokenizer(train_example)['input_ids']))
        else:
            max_len = max(max_len, len(model.tokenizer(train_example)['input_ids']))
        
    for t in train_prefixes:
        train_tokens.append(tokenizer.decode(model.tokenizer(t, padding='max_length', max_length=max_len)["input_ids"]))
    return train_tokens, max_len

if DATATYPE == 'conv-nonce':
    n_prompts, n_questions, n_golds = get_prompt_from_df(f'{NONCE_PROMPTS_PATH}/{nonceType}.csv')

prompts, questions, golds = get_prompt_from_df(f'{PROMPT_FILES_PATH}/{sType}.csv')
train_prefixes, max_len = getPaddedTrainTokens(prompts, golds)

if 'mistral-v.0.3' in MODEL_NAME:
    mlp_mean_cache = torch.zeros((model.config.num_hidden_layers, max_len, model.model.layers[0].self_attn.o_proj.out_features)).to("cuda")
    attn_mean_cache = torch.zeros((model.config.num_hidden_layers, max_len, model.model.layers[0].mlp.down_proj.out_features)).to("cuda")
elif 'llama-2-7b' in MODEL_NAME:
    mlp_mean_cache = torch.zeros((model.config.num_hidden_layers, max_len + 1, model.model.layers[0].self_attn.o_proj.out_features)).to("cuda")
    attn_mean_cache = torch.zeros((model.config.num_hidden_layers, max_len + 1, model.model.layers[0].mlp.down_proj.out_features)).to("cuda")
elif 'qwen' in MODEL_NAME:
    mlp_mean_cache = torch.zeros((model.config.num_hidden_layers, max_len - 1, model.model.layers[0].self_attn.o_proj.out_features)).to("cuda")
    attn_mean_cache = torch.zeros((model.config.num_hidden_layers, max_len - 1, model.model.layers[0].mlp.down_proj.out_features)).to("cuda")
else:
    mlp_mean_cache = torch.zeros((model.config.num_hidden_layers, max_len, model.model.layers[0].self_attn.o_proj.out_features)).to("cuda")
    attn_mean_cache = torch.zeros((model.config.num_hidden_layers, max_len, model.model.layers[0].mlp.down_proj.out_features)).to("cuda")
for tr_prefix in tqdm(train_prefixes):
    with model.trace(tr_prefix.rstrip(), scan=False, validate=False) as tracer:
        for layer in range(len(model.model.layers)):
            self_attn = model.model.layers[layer].self_attn.o_proj.output
            mlp = model.model.layers[layer].mlp.down_proj.output
            attn_mean_cache[layer] += self_attn[0,:,:].detach().save()
            mlp_mean_cache[layer] += mlp[0,:,:].detach().save()

attn_mean_cache /= len(train_prefixes)
mlp_mean_cache /= len(train_prefixes)


if not DATATYPE == 'conv-nonce':
    print(f"Writing to {PATCH_PICKLES_PATH}/mlp/{PATCH_PICKLES_SUBPATH}/mean-{sType}.pkl")
    with open(f'{PATCH_PICKLES_PATH}/mlp/{PATCH_PICKLES_SUBPATH}/mean-{sType}.pkl', 'wb') as f:
        
        pickle.dump(mlp_mean_cache[:, :, :], f)

    with open(f'{PATCH_PICKLES_PATH}/attn/{PATCH_PICKLES_SUBPATH}/mean-{sType}.pkl', 'wb') as f:
        pickle.dump(attn_mean_cache[:, :, :], f)
else:
    print(f"Writing to {PATCH_PICKLES_PATH}/mlp/{PATCH_PICKLES_SUBPATH}/mean-{nonceType}-conv-nonce-shaped.pkl")
    with open(f'{PATCH_PICKLES_PATH}/mlp/{PATCH_PICKLES_SUBPATH}/mean-{nonceType}-conv-nonce-shaped.pkl', 'wb') as f:
        pickle.dump(mlp_mean_cache[:, :, :], f)

    with open(f'{PATCH_PICKLES_PATH}/attn/{PATCH_PICKLES_SUBPATH}/mean-{nonceType}-conv-nonce-shaped.pkl', 'wb') as f:
        pickle.dump(attn_mean_cache[:, :, :], f)