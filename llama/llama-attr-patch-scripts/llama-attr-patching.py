from nnsight import LanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer
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
PREFIX = '/share/u/arunas'
og = pd.read_csv(f'{PREFIX}/broca/ngs.csv')
print(og.columns)

model = LanguageModel("meta-llama/Llama-2-70b-hf",  load_in_4bit=True, dispatch=True, device_map='auto') # Load the model

def get_prompt_from_df(filename):
    data = list(pd.read_csv(filename)['prompt'])
    data = [sentence.strip() for sentence in data]
    data = [sentence for sentence in data if not sentence == '']
    data = [sentence.replace('</s>', '\n') for sentence in data]
    golds = [sentence.strip().split("\n")[-1].strip().split('A:')[-1].strip() for sentence in data]
    data = [sentence[: -len(golds[idx])].strip() for idx, sentence in enumerate(data)]
    return data, golds

types = ['it', 'jp-r-2-passive', 'it-r-1-null_subject', 'jp-r-3-subordinate', 'it-r-2-passive', 'jp-u-1-negation', 'it-r-3-subordinate', 'jp-u-2-invert', 'it-u-1-negation', 'jp-u-3-past-tense', 'it-u-2-invert', 'passive-sentence', 'it-u-3-gender', 'sentence', 'jp-r-1-sov', 'subordinate-sentence']

def callWithStype(sType):
    prompts, golds = get_prompt_from_df(f'{PREFIX}/broca/llama/llama-prompt-outputs/llama-classification-train-test-det-{sType}-new.csv')

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

    with open(f'{PREFIX}/broca/llama/llama-attr-patch-scripts/mlp/{sType}.pkl', 'wb') as f:
        pickle.dump(two_d_indices, f)

    flattened_effects_cache = attn_effects_cache.view(-1)
    top_neurons = flattened_effects_cache.topk(k=40)
    two_d_indices = torch.cat((((top_neurons[1] // attn_effects_cache.shape[1]).unsqueeze(1)), ((top_neurons[1] % attn_effects_cache.shape[1]).unsqueeze(1))), dim=1)

    with open(f'{PREFIX}/broca/llama/llama-attr-patch-scripts/attn/{sType}.pkl', 'wb') as f:
        pickle.dump(two_d_indices, f)

for sType in types:
    callWithStype(sType)
