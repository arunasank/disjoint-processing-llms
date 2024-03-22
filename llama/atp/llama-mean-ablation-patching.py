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

types = [col for col in list(og.columns) if not 'ng-' in col] 

def getMaxTokenLenAndPrefixes(prompts, golds):
    max_len = 0
    test_prefixes = []
    train_examples = set()
    train_prefixes = []
    for prompt in prompts:
        testExample = [x.strip() for m in prompt.split("Q: Is this sentence grammatical? Yes or No: ") for x in m.strip().split("A: ") if len(x.strip()) > 3]
        # t = [x.strip() for m in prompt.split("Q: Is this sentence grammatical? Yes or No: ") for x in m.strip().split("A: ") if len(x.strip()) > 3]
        # train_examples.update(t[:-1])
        test_prefixes.append(model.tokenizer(t[-1])["input_ids"])
        max_len = max(max_len, len(test_prefixes[-1]))
        
    for t in train_prefixes:
        train_prefixes.append(model.tokenizer(t)["input_ids"])
        
    return max_len, test_prefixes, train_prefixes
        

prompts, golds = get_prompt_from_df(f'{PREFIX}/broca/llama/experiments/llama-classification-train-test-det-{sType}.csv')
    trainPrefixes, max_len = getMaxTokenizedLen(prompts)
    mlp_mean_cache = torch.zeros((model.config.num_hidden_layers, model.config.hidden_size)).to("cuda")
    attn_mean_cache = torch.zeros((model.config.num_hidden_layers, model.config.hidden_size)).to("cuda")

    promptPrefix = fullPrompt[fullPrompt[:-2].rfind(':')+1:-2].strip()
    goldToken = model.tokenizer(gold)["input_ids"]
    promptPrefixToken = model.tokenizer(promptPrefix, max_len=max_len, padding=True)["input_ids"]
    
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
        
            mlp_mean_cache[layer] += mlp_effects[0].to(mlp_mean_cache[layer].device)
            attn_mean_cache[layer] += attn_effects[0].to(attn_mean_cache[layer].device)


    for prompt,gold in tqdm(zip(prompts, golds)):
        try:
            attrPatching(prompt, gold)
        except:
            print(f"Error with stype: {sType} prompt: {prompt} gold: {gold}", traceback.format_exc())
            continue

    mlp_mean_cache /= len(prompts)
    attn_mean_cache /= len(prompts)

    mlp_mean_cache = torch.nan_to_num(mlp_mean_cache)
    attn_mean_cache = torch.nan_to_num(attn_mean_cache)

    flattened_effects_cache = mlp_mean_cache.view(-1)
    top_neurons = flattened_effects_cache.topk(k=40)
    two_d_indices = torch.cat((((top_neurons[1] // mlp_mean_cache.shape[1]).unsqueeze(1)), ((top_neurons[1] % mlp_mean_cache.shape[1]).unsqueeze(1))), dim=1)

    with open(f'{PREFIX}/broca/llama/llama-attr-patch-scripts/mlp/{sType}-new.pkl', 'wb') as f:
        pickle.dump(two_d_indices, f)

    flattened_effects_cache = attn_mean_cache.view(-1)
    top_neurons = flattened_effects_cache.topk(k=40)
    two_d_indices = torch.cat((((top_neurons[1] // attn_mean_cache.shape[1]).unsqueeze(1)), ((top_neurons[1] % attn_mean_cache.shape[1]).unsqueeze(1))), dim=1)

    with open(f'{PREFIX}/broca/llama/llama-attr-patch-scripts/attn/{sType}-new.pkl', 'wb') as f:
        pickle.dump(two_d_indices, f)

for sType in reversed(types):
    if not os.path.exists(f'{PREFIX}/broca/llama/llama-attr-patch-scripts/attn/{sType}-new.pkl') or not os.path.exists(f'{PREFIX}/broca/llama/llama-attr-patch-scripts/mlp/{sType}-new.pkl'):
        print(f'Calling {sType}')
        callWithStype(sType)
