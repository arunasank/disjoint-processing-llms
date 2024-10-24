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

os.environ['HF_HOME'] = "/mnt/align4_drive/data/huggingface"

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

df = pd.read_csv(f'{DATA_PATH}')
# gCols = [col for col in sorted(df.columns) if (('en' in col[:2]) or ('ita' in col[:3]) or ('jap' in col[:3])) and (not 'qsub' in col) and (not 'null_subject' in col)]
gCols = [col for col in sorted(df.columns) if not ('ng-' in col) and '_S' in col]
print(gCols, len(gCols))

col = gCols[args.stype]
# col = 'en_S-r-1'

print('######### ', col)

if (ABLATION):
    print('ABLATION!')
    if (MODEL_NAME == 'mistral'):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, device_map="auto", padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        model = LanguageModel(MODEL_PATH, quantization_config=nf4_config, device_map='auto') # Load the model
        device_map = infer_auto_device_map(model)
    elif (MODEL_NAME == 'llama'):
        MODEL_CACHE_PATH = config["model_cache_path"]
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, device_map="auto", padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        model = LanguageModel(MODEL_PATH, device_map='auto', quantization_config=nf4_config, cache_dir=MODEL_CACHE_PATH)
        device_map = infer_auto_device_map(model)

    PATCH_PICKLES_PATH = config["patch_pickles_path"]
    PATCH_PICKLES_SUBPATH = config["patch_pickles_subpath"]
    TOPK = config['topk']
    RANDOMLY_SAMPLE = config['randomly_sample']
    # ABLATE_UNION = config['ablate_union']
    ABLATE_INTERSECTION = config['ablate_intersection']
    NUM_HIDDEN_STATES = config['num_hidden_states']
    REAL = config['real']
    
    def retrieve_topK(col, component, topK):
        with open(f'{PATCH_PICKLES_PATH}/{component}/{PATCH_PICKLES_SUBPATH}/{col}.pkl', 'rb') as f:
            component_cache = pickle.load(f)
            component_cache = component_cache.cpu()
            flattened_effects_cache = component_cache.view(-1)
            top_neurons = flattened_effects_cache.topk(k=int(topK * flattened_effects_cache.shape[-1]), sorted=True)
            two_d_indices = torch.cat(((top_neurons[1] // component_cache.shape[1]).unsqueeze(1), (top_neurons[1] % component_cache.shape[1]).unsqueeze(1)), dim=1)
            # print(two_d_indices.shape)
            df = pd.DataFrame(torch.cat((two_d_indices, top_neurons[0].unsqueeze(1)), dim=1).numpy(), columns=['layer', 'neuron','value'])   
        return df
    
    def retrieve_randomK(col, component, TOPK, real):
        df = retrieve_intersection_super(component, TOPK, real)
        layer_counts = df['layer'].value_counts()
        random_df = pd.DataFrame(columns=['layer', 'neuron'])
        for layer, count in layer_counts.items():
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
                    if (not real and ('_S' in c) and ('-r-' in c)) or (real and ('_S' in c) and ('-u-' in c)):
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
                for _, c in enumerate(gCols):
                    if (not real and ('_S' in c) and ('-r-' in c)) or (real and ('_S' in c) and ('-u-' in c)):
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
                # print(col, ' pre-deletion ', len(int_df))             
                for layer in real_df['layer'].unique():
                    real_language_neuron_count = len(real_df[real_df['layer'] == layer])
                    unreal_language_neuron_count = len(int_df[int_df['layer'] == layer])
                    difference = unreal_language_neuron_count - real_language_neuron_count
                    if ( difference > 0 ):
                        int_df.drop(int_df[int_df['layer'] == layer].tail(difference).index, inplace=True)
                # print(col, ' post-deletion ', len(int_df))             
                
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
        with open(f'{PATCH_PICKLES_PATH}/{component}/{PATCH_PICKLES_SUBPATH}/mean-{col}.pkl', 'rb') as mf:
            component_cache = pickle.load(mf)
            component_cache = component_cache.cpu()
            comp_values = []
            # print('################ Component_Cache shape', component_cache.shape)
            # print(component_cache)
            for _, row in df.iterrows():
                comp_values.append(list(component_cache[row['layer'], :, row['neuron']].numpy().flatten()))
            MAX_LEN = component_cache.shape[1]
            df['values'] = comp_values
        return df

    with torch.no_grad():
        mlp_ablate = ablation_cache(col, 'mlp')
        attn_ablate = ablation_cache(col, 'attn')
else:
    if (MODEL_NAME == 'mistral'):
        model_config = AutoConfig.from_pretrained(MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, config=model_config, device_map="auto", padding_side="left")
        
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=nf4_config, config=model_config, device_map='auto') # Load the model
    
    elif (MODEL_NAME == 'llama'):
        MODEL_CACHE_PATH = config["model_cache_path"]
        model_config = AutoConfig.from_pretrained(MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, device_map="auto", padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=nf4_config, device_map='auto', cache_dir=MODEL_CACHE_PATH)
    else:
        raise Exception("Model is neither llama nor mistral!")
    device_map = infer_auto_device_map(model)

def parse_answer(text):
    answers = []
    for t in text:
        ans = t.split("A: ")[-1].strip()
        answers.append(ans)
        # answers.append(t)
    return answers

def compute_accuracy(preds, golds):
    # print(len(preds), len(golds))
    print(preds)
    assert len(preds) == len(golds), f"Predictions and golds must have the same length {len(preds)}, {len(golds)}"
    total = 0
    correct = 0
    for pred, gold in zip(preds, golds):
        if pred == gold:
            correct += 1
        total += 1
    return correct / total


@torch.no_grad()
def get_aligned_words_measures(texts, 
                               answers,
                               measure,
                               model, 
                               tokenizer) -> list[str]:
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
        yes_token_id = tokenizer(" Yes")['input_ids']
        no_token_id = tokenizer(" No")['input_ids']
        # print("TOKENS ", yes_token_id, no_token_id)
        # print("DECODE ", tokenizer.decode(yes_token_id), tokenizer.decode(no_token_id))
        if (ABLATION):
            # # print('# MAX LENGTH ', MAX_LEN)
            fPrompts = tokenizer.batch_decode(tokenizer(fPrompts, padding='max_length', max_length=MAX_LEN)["input_ids"])
            with model.trace(fPrompts, scan=False, validate=False) as tracer:
                for idx, row in mlp_ablate.iterrows():
                    model.model.layers[row['layer']].mlp.down_proj.output[:, :MAX_LEN, row['neuron']] = torch.tensor(row['values'])
                for idx, row in attn_ablate.iterrows():
                    model.model.layers[row['layer']].self_attn.o_proj.output[:, :MAX_LEN, row['neuron']] = torch.tensor(row['values'])
                token_ids = model.lm_head.output.save()
            # answers = model.tokenizer.batch_decode()
            yes_answers = token_ids.value[:,-1, yes_token_id].sum(dim=-1)
            no_answers = token_ids.value[:,-1, no_token_id].sum(dim=-1)
            answer_ids = torch.where(yes_answers > no_answers, 0, torch.where(yes_answers < no_answers, 1, -1))
            answers = ["Yes" if idx == 0 else "No" if idx == 1 else "Equal" for idx in answer_ids.tolist()]
            preds = preds + answers
            fPredictions = answers
            for batch_idx in range(len(fPrompts)):
                f = pd.concat([f, pd.DataFrame([{'type': col, 'prompt': fPrompts[batch_idx], 'q' :fQs[batch_idx], 'prediction': fPredictions[batch_idx], 'gold': fGolds[batch_idx], 'int-grad': 0}])]).reset_index(drop=True)
        else:
            model_inputs = tokenizer(fPrompts, return_tensors="pt", padding=True).to('cuda')
            answers = model(**model_inputs, labels=model_inputs['input_ids'])            
            logits = answers[1]
            yes_answers = logits[:, -1, yes_token_id].sum(dim=-1)
            no_answers = logits[:, -1, no_token_id].sum(dim=-1)
            answer_ids = torch.where(yes_answers > no_answers, 0, torch.where(yes_answers < no_answers, 1, -1))
            answers = ["Yes" if idx == 0 else "No" if idx == 1 else "Equal" for idx in answer_ids.tolist()]
            preds = preds + answers
            fPredictions = answers
            fSurprisals = get_aligned_words_measures(fQs, answers, "surp", model, tokenizer)
            for batch_idx in range(len(fPrompts)):
                f = pd.concat([f, pd.DataFrame([{'type': col, 'prompt': fPrompts[batch_idx], 'q' : fQs[batch_idx], 'prediction': fPredictions[batch_idx], 'gold': fGolds[batch_idx], 'surprisal': fSurprisals[batch_idx], 'int-grad': 0}])]).reset_index(drop=True)
    # Evaluate
    accuracy = compute_accuracy(preds, golds)
    print(f"{col} -- Accuracy: {accuracy:.2f}\n")
    g = pd.concat([g, pd.DataFrame([{ 'trainType' : col, 'testType': col, 'accuracy': f"{accuracy:.2f}"}])])
    f.to_csv(f"{PREFIX}/broca/{MODEL_NAME}/experiments/{FINAL_CSV_SUBPATH}/{col}.csv")
    g.to_csv(f'{PREFIX}/broca/{MODEL_NAME}/experiments/{FINAL_CSV_SUBPATH}/{col}-acc.csv', index=False)