from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig,
    BitsAndBytesConfig
)
import bitsandbytes
from accelerate import infer_auto_device_map
from nnsight import LanguageModel
import sys
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import Dataset, load_dataset
import torch
import random
import pandas as pd
from tqdm import tqdm
import os
import pickle
import yaml
import argparse
import numpy as np
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

parser = argparse.ArgumentParser()

parser.add_argument('--config', type=str, help='path to the model training config file, found in broca/configs')
parser.add_argument('--stype', type=int, help='grammar structure col number, found in broca/data-gen')

args = parser.parse_args()
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

# args = { "config": "/mnt/align4_drive/arunas/broca/configs/mistral-icl-config", "stype": 7 }
# with open(args["config"], 'r') as f:
#     config = yaml.safe_load(f)

PREFIX = config["prefix"]
MODEL_NAME = config["model_name"]
MODEL_PATH = config["model_path"]
ABLATION = config["ablation"]
DATA_PATH = config["data_path"]
PROMPTS_PATH = config["prompts_path"]
NUM_DEMONSTRATIONS = config["num_dems"]
BATCH_SIZE = config["batch_size"]
FINAL_CSV_SUBPATH = config["final_csv_subpath"]
HF_TOKEN = os.environ.get('HF_TOKEN')
MAX_LEN = 0

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

df = pd.read_csv(f'{DATA_PATH}')
gCols = [col for col in sorted(df.columns) if (('en' in col[:2]) or ('ita' in col[:3]) or ('jap' in col[:3])) and (not 'qsub' in col) and (not 'null_subject' in col)]
print(gCols, len(gCols))

col = gCols[args.stype]

print('######### ', col)

if (ABLATION):
    print('ABLATION!')
    # Mistral
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, device_map="auto", padding_side="left")
    # tokenizer.pad_token = tokenizer.eos_token
    # model = LanguageModel(MODEL_PATH, quantization_config=nf4_config, device_map='auto') # Load the model
    # device_map = infer_auto_device_map(model)
    
    # Llama
    
    MODEL_CACHE_PATH = config["model_cache_path"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, device_map="auto", padding_side="left", cache_dir=MODEL_CACHE_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    model = LanguageModel(MODEL_PATH, device_map='auto', quantization_config=nf4_config, cache_dir=MODEL_CACHE_PATH)
    device_map = infer_auto_device_map(model)

    MEAN_PICKLES_PATH = config["mean_pickles_path"]
    MEAN_PICKLES_SUBPATH = config["mean_pickles_subpath"]
    PATCH_PICKLES_PATH = config["patch_pickles_path"]
    PATCH_PICKLES_SUBPATH = config["patch_pickles_subpath"]
    TOPK = config['topk']
    RANDOMLY_SAMPLE = config['randomly_sample']
    ABLATE_UNION = config['ablate_union']
    ABLATE_INTERSECTION = config['ablate_intersection']
    NUM_HIDDEN_STATES = config['num_hidden_states']
    REAL = config['real']
    
    def retrieve_topK(col, component, topK):
        with open(f'{PATCH_PICKLES_PATH}/{component}/{PATCH_PICKLES_SUBPATH}/{col}.pkl', 'rb') as f:
            component_cache = pickle.load(f)
            component_cache = component_cache.cpu()
            flattened_effects_cache = component_cache.view(-1)
            top_neurons = flattened_effects_cache.topk(k=int(topK * flattened_effects_cache.shape[-1]), sorted=True)
            two_d_indices = torch.cat((((top_neurons[1] // component_cache.shape[1]).unsqueeze(1)), ((top_neurons[1] % component_cache.shape[1]).unsqueeze(1))), dim=1)
            print(two_d_indices.shape)
            df = pd.DataFrame(torch.cat((two_d_indices, top_neurons[0].unsqueeze(1)), dim=1).numpy(), columns=['layer', 'neuron','value'])   
        return df
    
    def retrieve_randomK(col, component, TOPK, real):
        df = retrieve_intersection_super(component, TOPK, real)
        # udf = retrieve_intersection_super(component, TOPK, not real)
        # layer_counts = pd.DataFrame(df['layer'].value_counts()).reset_index()
        # u_layer_counts = pd.DataFrame(udf['layer'].value_counts()).reset_index()
        # all_layers = set(layer_counts['layer']).union(set(u_layer_counts['layer']))
        # layer_counts = layer_counts.set_index('layer').reindex(all_layers, fill_value=0)
        # u_layer_counts = u_layer_counts.set_index('layer').reindex(all_layers, fill_value=0)
        # max_counts = layer_counts['count'].combine(u_layer_counts['count'], max)

        layer_counts = df['layer'].value_counts()
        random_df = pd.DataFrame(columns=['layer', 'neuron'])
        # for layer, count in max_counts.items():
        for layer, count in layer_counts.items():
            # print(layer, count)
            max_neuron_index = NUM_HIDDEN_STATES
            sampled_neurons = np.random.choice(max_neuron_index, size=count, replace=False)
            temp_df = pd.DataFrame({'layer': [layer] * count, 'neuron': sampled_neurons})
            random_df = pd.concat([random_df, temp_df], ignore_index=True)
        with open(f'{FINAL_CSV_SUBPATH}.txt', 'a+') as file:
            file.write(f"{col} {len(random_df)}")
        return random_df

    def retrieve_intersection_super(component, TOPK, real):
        tuples = set()
        if (real):
            print('######## REAL INTERSECTION SUPER')
            if os.path.exists(f'{PATCH_PICKLES_PATH}/{component}/{PATCH_PICKLES_SUBPATH}/r-diagonal.csv'):
                int_df = pd.read_csv(f'{PATCH_PICKLES_PATH}/{component}/{PATCH_PICKLES_SUBPATH}/r-diagonal.csv')   
                return int_df
            else:
                for idx, c in enumerate(gCols):
                    if (not real and not '-u-' in c) or (real and '-u-' in c):
                        continue
                    df = retrieve_topK(c, component, TOPK)
                    cTuples = set([(a,b) for (a,b,c) in df.to_numpy()])
                    tuples = set.union(tuples, cTuples)
                int_df = pd.DataFrame(tuples, columns =['layer', 'neuron'])
                int_df[['layer', 'neuron']].astype(int).to_csv(f'{PATCH_PICKLES_PATH}/{component}/{PATCH_PICKLES_SUBPATH}/r-diagonal.csv', index=False)
        else:
            print('######## UNREAL INTERSECTION SUPER')
            if (os.path.exists(f'{PATCH_PICKLES_PATH}/{component}/{PATCH_PICKLES_SUBPATH}/u-diagonal.csv')):
                int_df = pd.read_csv(f'{PATCH_PICKLES_PATH}/{component}/{PATCH_PICKLES_SUBPATH}/u-diagonal.csv')    
                return int_df
            else:
                real_df = pd.read_csv(f'{PATCH_PICKLES_PATH}/{component}/{PATCH_PICKLES_SUBPATH}/r-diagonal.csv')
                for idx, c in enumerate(gCols):
                    if (not real and not '-u-' in c) or (real and '-u-' in c):
                        continue
                    df = retrieve_topK(c, component, TOPK)
                    cTuples = set([tuple(r) for r in df.to_numpy()])
                    
                    preferred_tuples = {}
                    for a, b, c in tuples:
                        if (a, b) not in preferred_tuples or c > preferred_tuples[(a, b)][2]:
                            preferred_tuples[(a, b)] = (a, b, c)
                    
                    for a, b, c in cTuples:
                        if (a, b) not in preferred_tuples or c > preferred_tuples[(a, b)][2]:
                            preferred_tuples[(a, b)] = (a, b, c)

                    tuples = set(preferred_tuples.values())
                int_df = pd.DataFrame(tuples, columns =['layer', 'neuron', 'value'])
                int_df.sort_values(['layer', 'neuron','value'], ascending=[True, True, False], inplace=True)
                print(col, ' pre-deletion ', len(int_df))             
                for layer in real_df['layer'].unique():
                    real_language_neuron_count = len(real_df[real_df['layer'] == layer])
                    unreal_language_neuron_count = len(int_df[int_df['layer'] == layer])
                    difference = unreal_language_neuron_count - real_language_neuron_count
                    if ( difference > 0 ):
                        int_df.drop(int_df[int_df['layer'] == layer].tail(difference).index, inplace=True)
                print(col, ' post-deletion ', len(int_df))             
                
                int_df[['layer', 'neuron']].astype(int).to_csv(f'{PATCH_PICKLES_PATH}/{component}/{PATCH_PICKLES_SUBPATH}/u-diagonal.csv', index=False)
        with open(f'{FINAL_CSV_SUBPATH}.txt', 'a+') as file:
            file.write(f"REAL={real} TOPK={TOPK} COMPONENT={component} LENGTH={len(int_df)}")
        return int_df[['layer', 'neuron']].astype(int)
    
    def ablation_cache(col, component):
        global MAX_LEN
        if (RANDOMLY_SAMPLE):
            print('### RANDOMLY SAMPLE NEURONS FOR ABLATION ')
            df = retrieve_randomK(col, component, TOPK, REAL)
        elif (ABLATE_INTERSECTION):
            print('### ABLATE INTERSECTION OF TOPK NEURONS FROM ALL LANGUAGES ')
            df = retrieve_intersection_super(component, TOPK, REAL)
        else:
            df = retrieve_topK(col, component, TOPK)
        with open(f'{MEAN_PICKLES_PATH}/{component}/{MEAN_PICKLES_SUBPATH}/mean-{col}.pkl', 'rb') as mf:
            component_cache = pickle.load(mf)
            component_cache = component_cache.cpu()
            comp_values = []
            print('################ Component_Cache shape', component_cache.shape)
            # print(component_cache)
            for _, row in df.iterrows():
                comp_values.append(list(component_cache[row['layer'], row['neuron']].numpy().flatten()))
            MAX_LEN = len(comp_values[0])
            df['values'] = comp_values
        return df

    with torch.no_grad():
        mlp_ablate = ablation_cache(col, 'mlp')
        attn_ablate = ablation_cache(col, 'attn')
else:
    # model_config = AutoConfig.from_pretrained(MODEL_PATH)
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, config=model_config, device_map="auto", padding_side="left")
    
    # tokenizer.pad_token = tokenizer.eos_token
    # model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=nf4_config, config=model_config, device_map='auto') # Load the model
    
    # device_map = infer_auto_device_map(model)
    
    
    # MODEL_CACHE_PATH = config["model_cache_path"]
    # model_config = AutoConfig.from_pretrained(MODEL_PATH)
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, device_map="auto", padding_side="left", cache_dir=MODEL_CACHE_PATH)
    # tokenizer.pad_token = tokenizer.eos_token
    # model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=nf4_config, device_map='auto', cache_dir=MODEL_CACHE_PATH)
    device_map = infer_auto_device_map(model)

def parse_answer(text):
    answers = []
    for t in text:
        ans = t.split("A: ")[-1].strip()
        answers.append(ans)
        # answers.append(t)
    return answers

def compute_accuracy(preds, golds):
    print(len(preds), len(golds))
    assert len(preds) == len(golds)
    total = 0
    correct = 0
    for pred, gold in zip(preds, golds):
        if pred == gold:
            correct += 1
        total += 1
    return correct / total


@torch.no_grad()
def get_aligned_words_measures(texts: str, 
                               answers: str,
                               measure: str,
                               model: GPT2LMHeadModel, 
                               tokenizer: GPT2Tokenizer) -> list[str]:
    if measure not in {'prob', 'surp'}:
        sys.stderr.write(f"{measure} not recognized\n")
        sys.exit(1)

    datas = []
    for t in range(len(texts)):
        text = f'{texts[t]} {answers[t]}'
        data = []
    
        ids = tokenizer(text, return_tensors='pt').to('cuda')
        input_ids = ids.input_ids.flatten().data
        target_ids = ids.input_ids[:,1:]
    
        # get output
        logits = model(**ids).logits
        output = torch.nn.functional.log_softmax(logits, dim=-1)
        if measure == 'surp':
            output = -(output/torch.log(torch.tensor(2.0)))
        else:
            output = torch.exp(output)
    
        # get by token measures 
        target_measures = output[:,:-1, :]
        # use gather to get the output for each target item in the batch
        target_measures = target_measures.gather(-1,
                                 target_ids.unsqueeze(2)).flatten().tolist()
        
        tokens = tokenizer.convert_ids_to_tokens(input_ids)[1:]
        words = text.split(' ')
    
        # A lil loop to force align words 
        current_word = words.pop(0)
        current_token = tokens.pop(0).replace('▁', '')
        measure = 0
        while len(data) != len(text.split(' ')) and len(target_measures) > 0:
            if current_word == current_token:
                data.append((current_word, measure))
                measure = 0
                if words:
                    current_word = words.pop(0)
                    current_token = tokens.pop(0).replace('▁', '')
                    measure += target_measures.pop(0)
            else:
                measure += target_measures.pop(0)
                current_token += tokens.pop(0).replace('▁', '')
                data.append((current_token, measure))
        datas.append(data)
    return datas

def get_prompt_from_df(filename):
    data = list(pd.read_csv(filename)['prompt'])
    questions = list(pd.read_csv(filename)['q'])
    golds = list(pd.read_csv(filename)['gold'])
    return data, questions, golds

preds = []
golds = []

f = pd.DataFrame(columns=["type", "prompt", "q", "prediction", "gold", "surprisal", "int-grad"])
f['type'] = 'test'
g = pd.DataFrame(columns=['accuracy', 'type'])

if (not (os.path.exists(f"{PREFIX}/broca/{MODEL_NAME}/experiments/{FINAL_CSV_SUBPATH}/{col}.csv")) and not (os.path.exists(f"{PREFIX}/broca/{MODEL_NAME}/experiments/{FINAL_CSV_SUBPATH}/{col}-acc.csv"))):
    prompts, questions, golds = get_prompt_from_df(f'{PROMPTS_PATH}/{col}.csv')
    f = pd.DataFrame(columns=["type", "prompt", "q", "prediction", "gold", "surprisal", "int-grad"])
    for i in tqdm(range(0, len(prompts), BATCH_SIZE)):
        fQs = []
        fGolds = []
        fPrompts = []
        for batch_idx in range(min(BATCH_SIZE, len(prompts) - i)):
            fGolds.append(golds[i + batch_idx])
            fPrompts.append(prompts[i + batch_idx].rstrip())
            fQs.append(questions[i + batch_idx])
        answers = []
        if (ABLATION):
            fPrompts = tokenizer.batch_decode(tokenizer(fPrompts, padding='max_length', max_length=MAX_LEN)["input_ids"])
            with model.trace(fPrompts, scan=False, validate=False) as tracer:
                for idx, row in mlp_ablate.iterrows():
                    model.model.layers[row['layer']].mlp.down_proj.output[:, :len(row['values']), row['neuron']] = torch.tensor(row['values'])
                for idx, row in attn_ablate.iterrows():
                    model.model.layers[row['layer']].self_attn.o_proj.output[:, :len(row['values']), row['neuron']] = torch.tensor(row['values'])
                token_ids = model.lm_head.output.argmax(dim=-1).save()
            answers = model.tokenizer.batch_decode(token_ids.value[:,-1])
            preds = preds + parse_answer(answers)
            fPredictions = parse_answer(answers)
            for batch_idx in range(len(fPrompts)):
                f = pd.concat([f, pd.DataFrame([{'type': col, 'prompt': fPrompts[batch_idx], 'q' :fQs[batch_idx], 'prediction': fPredictions[batch_idx], 'gold': fGolds[batch_idx], 'int-grad': 0}])]).reset_index(drop=True)
        else:
            model_inputs = tokenizer(fPrompts, return_tensors="pt", padding=True).to('cuda')
            answers = model.generate(**model_inputs, pad_token_id=tokenizer.eos_token_id, max_new_tokens=2, top_p=0.9, temperature=0.1, do_sample=True)            
            answers = tokenizer.batch_decode(answers)[:BATCH_SIZE]
            preds = preds + parse_answer(answers)
            fPredictions = parse_answer(answers)
            fSurprisals = get_aligned_words_measures(fQs, parse_answer(answers), "surp", model, tokenizer)
            for batch_idx in range(len(fPrompts)):
                f = pd.concat([f, pd.DataFrame([{'type': col, 'prompt': fPrompts[batch_idx], 'q' : fQs[batch_idx], 'prediction': fPredictions[batch_idx], 'gold': fGolds[batch_idx], 'surprisal': fSurprisals[batch_idx], 'int-grad': 0}])]).reset_index(drop=True)
    # Evaluate
    accuracy = compute_accuracy(preds, golds)
    print(f"{col} -- Accuracy: {accuracy:.2f}\n")
    g = pd.concat([g, pd.DataFrame([{ 'trainType' : col, 'testType': col, 'accuracy': f"{accuracy:.2f}"}])])
    f.to_csv(f"{PREFIX}/broca/{MODEL_NAME}/experiments/{FINAL_CSV_SUBPATH}/{col}.csv")
    g.to_csv(f'{PREFIX}/broca/{MODEL_NAME}/experiments/{FINAL_CSV_SUBPATH}/{col}-acc.csv', index=False)