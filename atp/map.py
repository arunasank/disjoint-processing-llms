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
parser.add_argument('--batch_size', type=int, required=False, default=8, help='batch size for the model')

args = parser.parse_args()
with open(args.config_file, 'r') as f:
    config_file = yaml.safe_load(f)

batch_size = args.batch_size
# print(json.dumps(config_file, indent=4))
PREFIX = config_file["prefix"]
MODEL_NAME = config_file["model_name"]
MODEL_PATH = config_file["model_path"]
DATA_PATH = config_file["data_path"]
PROMPT_FILES_PATH = config_file["prompt_files_path"]
PATCH_PICKLES_PATH = config_file["patch_pickles_path"]
PATCH_PICKLES_SUBPATH = config_file["patch_pickles_sub_path"]
DATATYPE = config_file["datatype"]

og = pd.read_csv(DATA_PATH)
types = []
# types = [col for col in sorted(og.columns) if (('en' in col[:2]) or ('ita' in col[:3]) or ('jap' in col[:3])) and (not 'qsub' in col) and (not 'null_subject' in col)]
if DATATYPE == "nonce":
    types = sorted([col for col in sorted(og.columns) if not ('ng-' in col) and '_S' in col and 'en_S-' in col])
elif DATATYPE == "conventional":
    types = sorted([col for col in og.columns \
            if not '_S' in col \
            and (not 'qsub' in col) and (not 'null_subject' in col) \
            and ('ita-' in col  or 'en-' in col or 'jap-' in col)])
sType = col = types[args.stype]
if (not os.path.exists(f'{PATCH_PICKLES_PATH}/attn/{PATCH_PICKLES_SUBPATH}/mean-{col}.pkl') or not os.path.exists(f'{PATCH_PICKLES_PATH}/mlp/{PATCH_PICKLES_SUBPATH}/mean-{col}.pkl')):
    config = AutoConfig.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, config=config, device_map="auto", padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    if '7b' in MODEL_NAME or '8b' in MODEL_NAME or '70b' in MODEL_NAME or 'v0.3' in MODEL_NAME:
        print(f"quantizing {MODEL_NAME}")
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = LanguageModel(MODEL_PATH, quantization_config=nf4_config, tokenizer=tokenizer, device_map='auto') # Load the model
    else:
        model = LanguageModel(MODEL_PATH, tokenizer=tokenizer, device_map='auto') # Load the model

    def get_prompt_from_df(filename):
        data = list(pd.read_csv(filename)['prompt'])
        questions = list(pd.read_csv(filename)['q'])
        golds = list(pd.read_csv(filename)['gold'])
        return data, questions, golds
    
    def getPaddedTrainTokens(prompts, golds):
        max_len = 0
        train_prefixes = []
        train_tokens = []
        for idx, prompt in enumerate(prompts):
            train_example = prompt
            train_prefixes.append(train_example)
            max_len = max(max_len, len(model.tokenizer(train_example)['input_ids']))
            
        for t in train_prefixes:
            train_tokens.append(tokenizer.decode(model.tokenizer(t, padding='max_length', max_length=max_len)["input_ids"]))
        return train_tokens, max_len
    
    
    prompts, questions, golds = get_prompt_from_df(f'{PROMPT_FILES_PATH}/{sType}.csv')
    prompts = [p.rstrip() for p in prompts]
    train_prefixes, max_len = getPaddedTrainTokens(prompts, golds)

    print("Num layers ", model.config.num_hidden_layers)
    print(model)
    if MODEL_NAME == 'gpt-2-xl':
        layers = model.transformer.h
        mlp_mean_cache = torch.zeros((model.config.num_hidden_layers, batch_size, max_len, len(layers[0].attn.resid_dropout.out_features))).to("cuda")
        attn_mean_cache = torch.zeros((model.config.num_hidden_layers, batch_size, max_len, len(layers[0].mlp.dropout.out_features))).to("cuda")
   
    else:
        layers = model.model.layers
        if MODEL_NAME == 'llama-2-7b':
            mlp_mean_cache = torch.zeros((model.config.num_hidden_layers, batch_size, max_len + 2, layers[0].self_attn.o_proj.out_features)).to("cuda")
            attn_mean_cache = torch.zeros((model.config.num_hidden_layers, batch_size, max_len + 2, layers[0].mlp.down_proj.out_features)).to("cuda")
        elif MODEL_NAME in ['llama-3.1-8b', 'mistral-v0.3']:
            mlp_mean_cache = torch.zeros((model.config.num_hidden_layers, batch_size, max_len + 1, layers[0].self_attn.o_proj.out_features)).to("cuda")
            attn_mean_cache = torch.zeros((model.config.num_hidden_layers, batch_size, max_len + 1, layers[0].mlp.down_proj.out_features)).to("cuda")
        else:
            mlp_mean_cache = torch.zeros((model.config.num_hidden_layers, batch_size, max_len, layers[0].self_attn.o_proj.out_features)).to("cuda")
            attn_mean_cache = torch.zeros((model.config.num_hidden_layers, batch_size, max_len, layers[0].mlp.down_proj.out_features)).to("cuda")
   
    
    for tr_prefix in tqdm(train_prefixes):
        with model.trace(tr_prefix, scan=False, validate=False) as tracer:
            for layer in range(len(layers)):
                if MODEL_NAME == 'gpt-2-xl':
                    self_attn = layers[layer].attn.c_proj.output
                    mlp = layers[layer].mlp.c_proj.output
                else:
                    self_attn = layers[layer].self_attn.o_proj.output
                    mlp = layers[layer].mlp.down_proj.output
                attn_mean_cache[layer] += self_attn[:,:,:].detach().save()
                mlp_mean_cache[layer] += mlp[:,:,:].detach().save()

    attn_mean_cache = torch.mean(attn_mean_cache, dim=1) 
    mlp_mean_cache = torch.mean(mlp_mean_cache, dim=1) 
    
    print(attn_mean_cache.shape)
    print(mlp_mean_cache.shape)
    print(f"Writing to {PATCH_PICKLES_PATH}/mlp/{PATCH_PICKLES_SUBPATH}/mean-{sType}.pkl")
    with open(f'{PATCH_PICKLES_PATH}/mlp/{PATCH_PICKLES_SUBPATH}/mean-{sType}.pkl', 'wb') as f:
        
        pickle.dump(mlp_mean_cache[:, :, :], f)
    
    with open(f'{PATCH_PICKLES_PATH}/attn/{PATCH_PICKLES_SUBPATH}/mean-{sType}.pkl', 'wb') as f:
        pickle.dump(attn_mean_cache[:, :, :], f)
