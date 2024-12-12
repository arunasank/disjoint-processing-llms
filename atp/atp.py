from nnsight import LanguageModel
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoConfig
import torch
import pandas as pd
from tqdm import tqdm
import pickle
import argparse
import yaml
import os
import json
import gc
parser = argparse.ArgumentParser()

parser.add_argument('--config_file', type=str, help='path to the model training config file, found in broca/config')
parser.add_argument('--stype', type=int, help='structure type idx. Can range from 0-30')
parser.add_argument('--batch_size', type=int, required=False, default=8, help='batch size for processing the data')

args = parser.parse_args()
with open(args.config_file, 'r') as f:
    config_file = yaml.safe_load(f)

# args = { "config_file": "/mnt/align4_drive/arunas/broca/configs/mistral-atp-config", "stype": 23 }
# with open(args["config_file"], 'r') as f:
#    config_file = yaml.safe_load(f)
print(json.dumps(config_file, indent=4))
PREFIX = config_file["prefix"]
MODEL_NAME = config_file["model_name"]
MODEL_PATH = config_file["model_path"]
DATA_PATH = config_file["data_path"]
PROMPT_FILES_PATH = config_file["prompt_files_path"]
PATCH_PICKLES_PATH = config_file["patch_pickles_path"]
PATCH_PICKLES_SUBPATH = config_file["patch_pickles_sub_path"]
DATATYPE = config_file["datatype"]
os.environ['HF_HOME'] = "/mnt/align4_drive/data/huggingface"
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

# Define batch size
batch_size = args.batch_size

og = pd.read_csv(DATA_PATH)
types = []
if DATATYPE == "nonce":
    types = sorted([col for col in sorted(og.columns) if not ('ng-' in col) and '_S' in col and 'en_S-' in col])
elif DATATYPE == "conventional":
    types = sorted([col for col in og.columns \
            if not '_S' in col \
            and (not 'qsub' in col) and (not 'null_subject' in col) \
            and ('ita-' in col  or 'en-' in col or 'jap-' in col)])
# types = sorted([col for col in og.columns if ('en' in col[:2] or 'jap' in col[:3] or 'ita' in col[:3]) and (not 'qsub' in col) and (not 'null_subject' in col)])
sType = types[args.stype]
print(types)
os.makedirs(f"{PATCH_PICKLES_PATH}/attn/{PATCH_PICKLES_SUBPATH}/", exist_ok=True)
os.makedirs(f"{PATCH_PICKLES_PATH}/mlp/{PATCH_PICKLES_SUBPATH}/", exist_ok=True)
if (not os.path.exists(f"{PATCH_PICKLES_PATH}/attn/{PATCH_PICKLES_SUBPATH}/{sType}.pkl") or not os.path.exists(f"{PATCH_PICKLES_PATH}/mlp/{PATCH_PICKLES_SUBPATH}/{sType}.pkl")):
    print(f"Running for {sType}")
    
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
    
    model.requires_grad_(True)

    print(model)
    def get_prompt_from_df(filename):
        data = list(pd.read_csv(filename)['prompt'])
        questions = list(pd.read_csv(filename)['q'])
        golds = list(pd.read_csv(filename)['gold'])
        return data, questions, golds

    mlp_effects_cache = torch.zeros((model.config.num_hidden_layers, batch_size, model.config.hidden_size)).to("cuda")
    attn_effects_cache = torch.zeros((model.config.num_hidden_layers, batch_size, model.config.hidden_size)).to("cuda")

    def attrPatching(cleanPrompt, q, gold, idx):
        attn_layer_cache_prompt = {}
        mlp_layer_cache_prompt = {}

        attn_layer_cache_patch = {}
        mlp_layer_cache_patch = {}

        if gold == 'Yes':
            patch = og[og[sType] == q][f"ng-{sType}"].head(1).item()
            patchPrompt = cleanPrompt.replace(q, patch)
            notGold = "No"
        else:
            patch = og[og[f"ng-{sType}"] == q][sType].head(1).item()
            patchPrompt = cleanPrompt.replace(q, patch)
            notGold = "Yes"

        if model.tokenizer(cleanPrompt, return_tensors="pt").input_ids.shape[-1] != \
            model.tokenizer(patchPrompt, return_tensors="pt").input_ids.shape[-1]:
            return

        gold = model.tokenizer(gold)["input_ids"]
        notGold = model.tokenizer(notGold)["input_ids"]
        with model.trace(cleanPrompt.rstrip(), scan=False, validate=False) as tracer:
            if MODEL_NAME == 'gpt-2-xl':
                layers = model.transformer.h
            else:
                layers = model.model.layers
            for layer in range(len(layers)):
                if MODEL_NAME == 'gpt-2-xl':
                    self_attn = layers[layer].attn.resid_dropout.output
                    mlp = layers[layer].mlp.dropout.output
                else:
                    self_attn = layers[layer].self_attn.o_proj.output
                    mlp = layers[layer].mlp.down_proj.output
                mlp.retain_grad()
                self_attn.retain_grad()

                attn_layer_cache_prompt[layer] = {"forward": self_attn.save()} # "backward": self_attn.grad.detach().save()}
                mlp_layer_cache_prompt[layer] = {"forward": mlp.save()}# "backward": mlp.grad.detach().save()}

            logits = model.lm_head.output.save()
        loss = logits.value[:, -1, notGold] - logits.value[:, -1, gold]
        loss = loss.sum()
        loss.backward()

        with model.trace(patchPrompt.rstrip(), scan=False, validate=False) as tracer:
            for layer in range(len(layers)):
                if MODEL_NAME == 'gpt-2-xl':
                    self_attn = layers[layer].attn.c_proj.output
                    mlp = layers[layer].mlp.c_proj.output
                else:
                    self_attn = layers[layer].self_attn.o_proj.output
                    mlp = layers[layer].mlp.down_proj.output

                attn_layer_cache_patch[layer] = {"forward": self_attn.save()}
                mlp_layer_cache_patch[layer] = {"forward": mlp.save()}

        for layer in range(len(layers)):
            mlp_effects = (mlp_layer_cache_prompt[layer]["forward"].value.grad * (mlp_layer_cache_patch[layer]["forward"].value - mlp_layer_cache_prompt[layer]["forward"].value)).detach()
            attn_effects = (attn_layer_cache_prompt[layer]["forward"].value.grad * (attn_layer_cache_patch[layer]["forward"].value - attn_layer_cache_prompt[layer]["forward"].value)).detach()

            mlp_effects = mlp_effects[:, -1, :] # batch, token, hidden_states
            attn_effects = attn_effects[:, -1, :] # batch, token, hidden_states

            mlp_effects_cache[layer] += mlp_effects.to(mlp_effects_cache[layer].get_device())
            attn_effects_cache[layer] += attn_effects.to(mlp_effects_cache[layer].get_device())

    prompts, questions, golds = get_prompt_from_df(f'{PROMPT_FILES_PATH}/{sType}.csv')
    
    # Split data into batches
    def batch_data(data, batch_size):
        for i in range(0, len(data), batch_size):
            yield data[i:i+batch_size]

    # Iterate over batches
    for batch_idx, batch in enumerate(tqdm(batch_data(list(zip(prompts, questions, golds)), batch_size))):
        batch_prompts, batch_questions, batch_golds = zip(*batch)
        
        # Process each element in the batch
        for idx, (prompt, q, gold) in enumerate(zip(batch_prompts, batch_questions, batch_golds)):
            attrPatching(prompt, q, gold, batch_idx * batch_size + idx)
        
        # Trigger garbage collection after each batch
        gc.collect()

    print(mlp_effects_cache.shape)
    print(attn_effects_cache.shape)
    mlp_effects_cache = torch.mean(mlp_effects_cache, dim=1)    
    attn_effects_cache = torch.mean(attn_effects_cache, dim=1)

    with open(f'{PATCH_PICKLES_PATH}/mlp/{PATCH_PICKLES_SUBPATH}/{sType}.pkl', 'wb') as f:
        pickle.dump(mlp_effects_cache, f)

    with open(f'{PATCH_PICKLES_PATH}/attn/{PATCH_PICKLES_SUBPATH}/{sType}.pkl', 'wb') as f:
        pickle.dump(attn_effects_cache, f)
