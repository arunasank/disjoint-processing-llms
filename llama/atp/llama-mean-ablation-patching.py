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

print(json.dumps(config_file, indent=4))
PREFIX = config_file["prefix"]
MODEL_NAME = config_file["model_name"]
MODEL_PATH = config_file["model_path"]
DATA_PATH = config_file["data_path"]
PROMPT_FILES_PATH = config_file["prompt_files_path"]
PATCH_PICKLES_PATH = config_file["patch_pickles_path"]
PATCH_PICKLES_SUBPATH = config_file["patch_pickles_sub_path"]

og = pd.read_csv(DATA_PATH)
types = [col for col in og.columns if not 'ng-' in col]

if (MODEL_NAME == "llama"):
    os.environ["HF_TOKEN"] = config_file["hf_token"]
    MODEL_CACHE_PATH = config_file["model_cache_path"]
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    config = AutoConfig.from_pretrained(MODEL_PATH, cache_dir=MODEL_CACHE_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, config=config, device_map="auto", padding_side="left", cache_dir=MODEL_CACHE_PATH)
    
    tokenizer.pad_token = tokenizer.eos_token
    model = LanguageModel(MODEL_PATH, quantization_config=nf4_config, tokenizer=tokenizer, device_map='auto', cache_dir=MODEL_CACHE_PATH) # Load the model

elif (MODEL_NAME == "mistral"):
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    config = AutoConfig.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, config=config, device_map="auto", padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = LanguageModel(MODEL_PATH, quantization_config=nf4_config, tokenizer=tokenizer, device_map='auto') # Load the model

def get_prompt_from_df(filename):
    data = list(pd.read_csv(filename)['prompt'])
    questions = list(pd.read_csv(filename)['q'])
    golds = list(pd.read_csv(filename)['gold'])
    return data, questions, golds

def getPaddedTrainTokens(prompts, questions, golds):
    max_len = 0
    test_prefixes = []
    train_prefixes = []
    train_tokens = []
    for idx, prompt in enumerate(prompts):
        if (MODEL_NAME == 'mistral'):
            train_example = prompt
        else:
            train_example = f"{prompt}Q: Is this sentence grammatical? Yes or No: {questions[idx]}\nA: " 
        train_prefixes.append(train_example)
        max_len = max(max_len, len(model.tokenizer(train_example)['input_ids']))
        
    for t in train_prefixes:
        train_tokens.append(tokenizer.decode(model.tokenizer(t, padding='max_length', max_length=max_len)["input_ids"]))
    return train_tokens, max_len

def callWithsType(sType):
    print(f'Calling {sType}')
    prompts, questions, golds = get_prompt_from_df(f'{PROMPT_FILES_PATH}/{sType}.csv')
    train_prefixes, max_len = getPaddedTrainTokens(prompts, questions, golds)

    mlp_mean_cache = torch.zeros((model.config.num_hidden_layers, max_len + 2, model.model.layers[0].self_attn.o_proj.out_features)).to("cuda")
    attn_mean_cache = torch.zeros((model.config.num_hidden_layers, max_len + 2, model.model.layers[0].mlp.down_proj.out_features)).to("cuda")

    for tr_prefix in tqdm(train_prefixes):
        with model.trace(tr_prefix, scan=False, validate=False) as tracer:
            for layer in range(len(model.model.layers)):
                self_attn = model.model.layers[layer].self_attn.o_proj.output
                mlp = model.model.layers[layer].mlp.down_proj.output
                attn_mean_cache[layer] += self_attn[0,:,:].detach().save()
                mlp_mean_cache[layer] += mlp[0,:,:].detach().save()

    attn_mean_cache /= len(train_prefixes)
    mlp_mean_cache /= len(train_prefixes)

    with open(f'{PATCH_PICKLES_PATH}/mlp/{PATCH_PICKLES_SUBPATH}/{sType}.pkl', 'wb') as f:
        pickle.dump(mlp_mean_cache[:, -1, :], f)
    
    with open(f'{PATCH_PICKLES_PATH}/attn/{PATCH_PICKLES_SUBPATH}/{sType}.pkl', 'wb') as f:
        pickle.dump(attn_mean_cache[:, -1, :], f)

#for sType in types[::-1]:
sType = types[args.stype]
try:
    if (not (os.path.exists(f"{PATCH_PICKLES_PATH}/mlp/{PATCH_PICKLES_SUBPATH}/{sType}.pkl")) or not (os.path.exists(f"{PATCH_PICKLES_PATH}/attn/{PATCH_PICKLES_SUBPATH}/{sType}.pkl"))):
        callWithsType(sType)
except Exception as e:
    print(f"Error with {sType}")
    print(e)
    traceback.print_exc()
print(f"Finished {sType}")
