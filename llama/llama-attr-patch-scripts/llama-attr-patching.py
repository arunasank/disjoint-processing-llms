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
os.environ['HF_TOKEN'] = "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"
PREFIX = '/mnt/align4_drive/arunas'
og = pd.read_csv(f'{PREFIX}/broca/data-gen/ngs.csv')
print(og.columns)

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

config = AutoConfig.from_pretrained("meta-llama/Llama-2-70b-hf", cache_dir='/mnt/align4_drive/arunas/llama-tensors/')
tokenizer = AutoTokenizer.from_pretrained(
           "meta-llama/Llama-2-70b-hf", config=config, device_map="auto", padding_side="left", cache_dir='/mnt/align4_drive/arunas/llama-tensors/'
           )

tokenizer.pad_token = tokenizer.eos_token

model = LanguageModel("meta-llama/Llama-2-70b-hf",  quantization_config=nf4_config, tokenizer=tokenizer, device_map='auto', cache_dir='/mnt/align4_drive/arunas/llama-tensors/') # Load the model


def get_prompt_from_df(filename):
    data = list(pd.read_csv(filename)['prompt'])
    data = [sentence.strip() for sentence in data]
    data = [sentence for sentence in data if not sentence == '']
    data = [sentence.replace('</s>', '\n') for sentence in data]
    golds = [sentence.strip().split("\n")[-1].strip().split('A:')[-1].strip() for sentence in data]
    data = [sentence[: -len(golds[idx])].strip() for idx, sentence in enumerate(data)]
    return data, golds

types = ['en', 'en-r-1-subordinate', 'en-r-2-passive', 'en-u-1-negation', 'en-u-2-inversion', 'en-u-3-qsubordinate', 'ita', 'ita-r-1-null_subject', 'ita-r-2-subordinate', 'ita-r-3-passive', 'ita-u-1-negation', 'ita-u-2-invert', 'ita-u-3-gender', 'it', 'it-r-1-null_subject', 'it-r-2-passive', 'it-r-3-subordinate', 'it-u-1-negation', 'it-u-2-invert', 'it-u-3-gender', 'jp-r-1-sov', 'jp-r-2-passive', 'jp-r-3-subordinate', 'jp-u-1-negation', 'jp-u-2-invert', 'jp-u-3-past-tense']

def callWithStype(sType):
    prompts, golds = get_prompt_from_df(f'{PREFIX}/broca/llama/experiments/llama-classification-train-test-det-{sType}.csv')

    mlp_effects_cache = torch.zeros((model.config.num_hidden_layers, model.config.hidden_size)).to("cuda")
    attn_effects_cache = torch.zeros((model.config.num_hidden_layers, model.config.hidden_size)).to("cuda")

    def attrPatching(fullPrompt, gold):
        attn_layer_cache_prompt = {}
        mlp_layer_cache_prompt = {}
        
        attn_layer_cache_patch = {}
        mlp_layer_cache_patch = {}
        if (gold == 'Yes'):
            predictionExample = fullPrompt[fullPrompt[:-2].rfind(':')+1:-2].strip()
            patch = og[og[sType] == predictionExample][f"ng-{sType}"].iloc[0]
            patchPrompt = fullPrompt.replace(predictionExample, patch)
        else:
            patchPrompt = fullPrompt
            patch = fullPrompt[fullPrompt[:-2].rfind(':')+1:-2].strip()
            predictionExample = og[og[f"ng-{sType}"] == patch][sType].iloc[0]
            fullPrompt = patchPrompt.replace(patch, predictionExample)
            gold = "Yes"

        notGold = "No"
        gold = model.tokenizer(gold)["input_ids"]
        notGold = model.tokenizer(notGold)["input_ids"]
        with model.trace(fullPrompt, scan=False, validate=False) as tracer:
            for layer in range(len(model.model.layers)):
                self_attn = model.model.layers[layer].self_attn.o_proj.output
                mlp = model.model.layers[layer].mlp.down_proj.output
            
                attn_layer_cache_prompt[layer] = {"forward": self_attn.detach().save(), "backward": self_attn.grad.detach().save()}
                mlp_layer_cache_prompt[layer] = {"forward": mlp.detach().save(), "backward": mlp.grad.detach().save()}
            
            logits = model.lm_head.output[:, -1, notGold] - model.lm_head.output[:, -1, gold]
            loss = logits.sum()
            loss.backward(retain_graph=False)
        
    
        with model.trace(patchPrompt, scan=False, validate=False) as tracer:
            for layer in range(len(model.model.layers)):
                self_attn = model.model.layers[layer].self_attn.o_proj.output
                mlp = model.model.layers[layer].mlp.down_proj.output
        
                attn_layer_cache_patch[layer] = {"forward": self_attn.detach().save()}
                mlp_layer_cache_patch[layer] = {"forward": mlp.detach().save()}
        
        for layer in range(len(model.model.layers)):
            mlp_effects = mlp_layer_cache_prompt[layer]["backward"].value * (mlp_layer_cache_patch[layer]["forward"].value - mlp_layer_cache_prompt[layer]["forward"].value)
            attn_effects = attn_layer_cache_prompt[layer]["backward"].value * (attn_layer_cache_patch[layer]["forward"].value - attn_layer_cache_prompt[layer]["forward"].value)
        
            mlp_effects = mlp_effects[:, -1, :] # batch, token, hidden_states
            attn_effects = attn_effects[:, -1, :] # batch, token, hidden_states
        
            mlp_effects_cache[layer] += mlp_effects[0].to(mlp_effects_cache[layer].device)
            attn_effects_cache[layer] += attn_effects[0].to(attn_effects_cache[layer].device)


    for prompt,gold in tqdm(zip(prompts, golds)):
        try:
            attrPatching(prompt, gold)
        except:
            print(f"Error with stype: {sType} prompt: {prompt} gold: {gold}", traceback.format_exc())
            continue

    mlp_effects_cache /= len(prompts)
    attn_effects_cache /= len(prompts)

    mlp_effects_cache = torch.nan_to_num(mlp_effects_cache)
    attn_effects_cache = torch.nan_to_num(attn_effects_cache)

    flattened_effects_cache = mlp_effects_cache.view(-1)
    top_neurons = flattened_effects_cache.topk(k=40)
    two_d_indices = torch.cat((((top_neurons[1] // mlp_effects_cache.shape[1]).unsqueeze(1)), ((top_neurons[1] % mlp_effects_cache.shape[1]).unsqueeze(1))), dim=1)

    with open(f'{PREFIX}/broca/llama/llama-attr-patch-scripts/mlp/{sType}-new.pkl', 'wb') as f:
        pickle.dump(two_d_indices, f)

    flattened_effects_cache = attn_effects_cache.view(-1)
    top_neurons = flattened_effects_cache.topk(k=40)
    two_d_indices = torch.cat((((top_neurons[1] // attn_effects_cache.shape[1]).unsqueeze(1)), ((top_neurons[1] % attn_effects_cache.shape[1]).unsqueeze(1))), dim=1)

    with open(f'{PREFIX}/broca/llama/llama-attr-patch-scripts/attn/{sType}-new.pkl', 'wb') as f:
        pickle.dump(two_d_indices, f)

for sType in types:
    callWithStype(sType)
