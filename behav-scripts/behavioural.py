import os
import sys
import random
import argparse
import pickle
from accelerate.utils import MODEL_NAME
from numpy.random import random_sample
import yaml
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
)
from accelerate import infer_auto_device_map
from nnsight import LanguageModel

# Set random seeds for reproducibility
def set_random_seeds(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
# Parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Model training script.')
    parser.add_argument('--config', type=str, help='Path to the model config file.')
    parser.add_argument('--stype', type=int, help='Column number for grammar structure.')
    return parser.parse_args()

# Load configuration from a YAML file
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
# Set environment variables
def set_environment():
    os.environ['HF_HOME'] = "/mnt/align4_drive/data/huggingface"
    os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
    
# Initialize tokenizer and model based on the configuration
def initialize_model():
    config = load_config(args.config)
    model_name, model_path = config["model_name"], config["model_path"]
    cache_dir = config.get("model_cache_path")
    ablation = config.get("ablation")
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, device_map="auto", padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token

    if model_name == 'mistral':
        if ablation:
            model = LanguageModel(model_path, 
                                  quantization_config=nf4_config, 
                                  device_map='auto', 
                                  cache_dir=cache_dir)
        else:
            model_config = AutoConfig.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                quantization_config=nf4_config, 
                config=model_config, 
                device_map='auto', 
                cache_dir=cache_dir
            )
    elif model_name == 'llama':
        if ablation:
            model = LanguageModel(model_path, 
                                  device_map='auto', 
                                  quantization_config=nf4_config, 
                                  cache_dir=cache_dir) 
        else:
            model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            quantization_config=nf4_config, 
            config=AutoConfig.from_pretrained(model_path), 
            device_map='auto', 
            cache_dir=cache_dir
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return tokenizer, model
    
# Compute accuracy between predictions and gold labels
def compute_accuracy(preds, golds):
    # print(preds, golds)
    assert len(preds) == len(golds), "Predictions and golds must be of the same length."
    correct = sum(1 for p, g in zip(preds, golds) if p == g)
    return correct / len(preds)

# Parse answers from text
def parse_answer(text):
    return [t.split("A: ")[-1].strip() for t in text]

# Generate dataset for experiments
def get_dataset_from_df(filename):
    data = pd.read_csv(filename)
    print(len(data['prompt']), len(data['q']), len(data['gold']))
    return list(data['prompt']), list(data['q']), list(data['gold'])

# Retrieve top K neurons for ablation
def retrieve_top_k_neurons(col, component):
    config = load_config(args.config)
    path = config.get("patch_pickles_path")
    subpath = config.get("patch_pickles_subpath")
    top_k = config.get("topk")
    comp_cache_file = f'{path}/{component}/{subpath}/{col}.pkl'
    with open(comp_cache_file, 'rb') as f:
        comp_cache = pickle.load(f).cpu()
        flattened_effects_cache = comp_cache.view(-1)
        top_neurons = flattened_effects_cache.topk(
            k=int(top_k * flattened_effects_cache.shape[-1]), 
            sorted=True
        )
        indices = torch.cat([
            (top_neurons[1] // comp_cache.shape[1]).unsqueeze(1),
            (top_neurons[1] % comp_cache.shape[1]).unsqueeze(1)
        ], dim=1)
        return pd.DataFrame(torch.cat((indices, top_neurons[0].unsqueeze(1)), 
                            dim=1).numpy(), columns=['layer', 'neuron', 'value'])

# Randomly sample K neurons for ablation
def retrieve_random_k(component):
    config = load_config(args.config)
    num_hidden_states = config.get("num_hidden_states")
    df = retrieve_union_top_k(component)
    layer_counts = df.groupby('layer')['neuron'].nunique()
    layer_counts = layer_counts.to_dict()
    random_df = pd.DataFrame(columns=['layer', 'neuron'])
    for layer in layer_counts:
        count = layer_counts[layer]
        max_neuron_index = num_hidden_states
        sampled_neurons = np.random.choice(max_neuron_index, size=count, replace=False)
        temp_df = pd.DataFrame({'layer': [layer] * count, 'neuron': sampled_neurons})
        random_df = pd.concat([random_df, temp_df], ignore_index=True)
    return random_df

def retrieve_union_top_k(component):
    config = load_config(args.config)
    path = config.get("patch_pickles_path")
    subpath = config.get("patch_pickles_subpath")
    real = config.get("real")
    datatype = config.get("datatype")
    prefix = config.get("prefix")
    model_name = config.get("model_name")
    # Load data and run the experiment
    data_path = config["data_path"]
    final_csv_subpath = config.get("final_csv_subpath")
    df = pd.read_csv(data_path)
    nonce = (datatype == "nonce")
    g_cols = []
    if datatype == "nonce":
        g_cols = [col for col in sorted(df.columns) \
            if not ('ng-' in col) and '_S' in col]
    elif datatype == "standard":
        g_cols = sorted([col for col in df.columns \
            if (not '_S' in col) and (not 'ng-' in col) \
            and (not 'qsub' in col) and (not 'null_subject' in col) \
            and ('ita-' in col  or 'en-' in col or 'jap-' in col)
        ])
    
    assert len(g_cols) > 0, "No columns found."
    def retrieve_union_set(nonce, real):
        if real:
            union_file_path = \
            f'{prefix}/broca/{model_name}/experiments/{final_csv_subpath}/r-diagonal.csv'
        else:
            union_file_path = \
            f'{prefix}/broca/{model_name}/experiments/{final_csv_subpath}/u-diagonal.csv'
        
        if os.path.exists(union_file_path):
            return pd.read_csv(union_file_path)
    
        union_set = set()
        for _, c in enumerate(g_cols):
            col_is_real = '-r-' in c
            nonce_col = '_S' in c
            if (nonce and not nonce_col) or (not nonce and nonce_col):
                continue
            if not (real == col_is_real):
                continue
            top_k_neurons = retrieve_top_k_neurons(c, component)
            l_n_v = []
            for layer, neuron, value in top_k_neurons.to_numpy():
                l_n_v.append((layer, neuron, value)) 
            l_n_v = set(l_n_v)
            union_set = set.union(union_set, l_n_v)
        union_set_df = pd.DataFrame(union_set, columns =['layer', 'neuron', 'value'])
        union_set_df = union_set_df.loc[union_set_df.groupby(['layer', 'neuron'])['value'].idxmax()]
        union_set_df[['layer', 'neuron']] = union_set_df[['layer', 'neuron']].astype(int)
        union_set_df.to_csv(union_file_path, index=False)
        return union_set_df.sort_values(by=['layer', 'neuron', 'value'], ascending=[True, True, False] )
    
    real_count = []
    unreal_count = []
    if (real):
        return retrieve_union_set(nonce, real=True)
    else:
        real_union_df = retrieve_union_set(nonce, real=True)
        unreal_union_df = retrieve_union_set(nonce, real=False)
        total_diff = []
        for layer in real_union_df['layer'].unique():
            real_language_neuron_count = len(real_union_df[real_union_df['layer'] == layer])
            unreal_language_neuron_count = len(unreal_union_df[unreal_union_df['layer'] == layer])
            real_count.append(real_language_neuron_count)
            unreal_count.append(unreal_language_neuron_count)
            difference = unreal_language_neuron_count - real_language_neuron_count
            if len(total_diff) == layer:
                total_diff.append(difference)
            else:
                while len(total_diff) < layer:
                    total_diff.append(0)
                    real_count.append(0)
                    unreal_count.append(0)
                total_diff.append(difference)
            if ( difference > 0 ):
                unreal_union_df.drop(unreal_union_df[unreal_union_df['layer'] == layer].tail(difference).index, inplace=True)
        with open(f'{prefix}/broca/{model_name}/experiments/{final_csv_subpath}/difference.txt', 'a') as f:
            for layer, diff in enumerate(total_diff):
                f.write(f"Layer {layer}: {diff}, {real_count[layer]}, {unreal_count[layer]}\n")
        unreal_union_df.to_csv(f'{prefix}/broca/{model_name}/experiments/{final_csv_subpath}/u-diagonal.csv', index=False)
        return unreal_union_df

def get_mean_ablation_vals(component, col):
    config = load_config(args.config)
    path = config.get("patch_pickles_path")
    subpath = config.get("patch_pickles_subpath")
    # which neurons to ablate?
    neurons_to_abl = ablate_neurons(component, col)    
    mean_ablations_filepath =  f'{path}/{component}/{subpath}/mean-{col}.pkl'
    
    with open(mean_ablations_filepath, 'rb') as f:
        mean_abl_file = pickle.load(f)
    
    mean_abl_file = mean_abl_file.detach().cpu()
    mean_vals = []
    for _, row in neurons_to_abl.iterrows():
        # print(row['layer'], row['neuron'])  
        mean_vals.append(list(mean_abl_file[row['layer'], :, row['neuron']].detach().cpu().flatten()))
    # set ablation_values
    neurons_to_abl['values'] = mean_vals
    return neurons_to_abl, mean_abl_file.shape[1]
        
# Perform ablation for model layers
def ablate_neurons(component, col):
    config = load_config(args.config)
    random_sample = config.get('random_ablate')
    union = config.get('intersection_ablate')
    if random_sample:
        return retrieve_random_k(component)[['layer', 'neuron']].astype(int)
    if union:
        print('intersection ablate ', config['real'])
        return retrieve_union_top_k(component)[['layer', 'neuron']].astype(int)
    # In all other cases, return top k neurons
    return retrieve_top_k_neurons(col, component)[['layer', 'neuron']].astype(int)

def run_ablation_experiment(config, col):
    print("Performing ablation...")
    batch_size = config.get("batch_size")
    final_csv_subpath = config.get("final_csv_subpath")
    prefix = config.get("prefix")
    model_name = config.get("model_name")
    mlp_ablate, max_prompt_len_mlp = get_mean_ablation_vals('mlp', col)
    attn_ablate, max_prompt_len_attn = get_mean_ablation_vals('attn', col)
    assert max_prompt_len_mlp == max_prompt_len_attn, "Prompt lengths must match."
    max_prompt_len = max_prompt_len_mlp
    
    tokenizer, model = initialize_model()
    
    file_store = pd.DataFrame(columns=["type", "prompt", "q", "prediction", "gold", "surprisal"])
    
    yes_token_id = tokenizer(" Yes")['input_ids']
    no_token_id = tokenizer(" No")['input_ids']
    
    prompts, questions, golds = get_dataset_from_df(f"{config['prompts_path']}/{col}.csv")
    prompts = [p.strip() for p in prompts]
    preds = []
    # Processing predictions
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i + batch_size]
        batch_golds = golds[i:i + batch_size]
        batch_questions = questions[i:i + batch_size]
        
        tokenizer_inputs = tokenizer(batch_prompts, padding='max_length', max_length=max_prompt_len, return_tensors="pt")
        with model.trace(tokenizer_inputs, labels=tokenizer_inputs['input_ids'], scan=False, validate=False) as _:
            for _, row in mlp_ablate.iterrows():
                # assert model.model.layers[row['layer']].mlp.down_proj.output.shape[1] == max_prompt_len
                model.model.layers[row['layer']].mlp.down_proj.output[:, :, row['neuron']] = torch.tensor(row['values'])
            for _, row in attn_ablate.iterrows():
                # assert model.model.layers[row['layer']].self_attn.o_proj.output.shape[1] == max_prompt_len
                model.model.layers[row['layer']].self_attn.o_proj.output[:, :, row['neuron']] = torch.tensor(row['values'])
            logits = model.lm_head.output.detach().cpu().save()
        
        yes_answers = logits.value[:,-1, yes_token_id][:, -1]
        no_answers = logits.value[:,-1, no_token_id][:, -1]
        answer_ids = torch.where(yes_answers > no_answers, 0, torch.where(yes_answers < no_answers, 1, -1))
        batch_predictions = ["Yes" if idx == 0 else "No" if idx == 1 else "Equal" for idx in answer_ids.tolist()]
        preds = preds + batch_predictions
        # print(preds, batch_golds)
        batch_surprisals = get_surprisals(logits, tokenizer_inputs['input_ids'], tokenizer.eos_token_id)
        
        for idx in range(len(batch_prompts)):
            batch_file_store = pd.DataFrame([{'type': col, 
                'prompt': batch_prompts[idx], 
                'q' :batch_questions[idx], 
                'prediction': batch_predictions[idx], 
                'gold': batch_golds[idx],
                'surprisal': batch_surprisals[idx]
            }])
            file_store = pd.concat([file_store, batch_file_store]).reset_index(drop=True)
    file_store.to_csv(f"{prefix}/broca/{model_name}/experiments/{final_csv_subpath}/{col}.csv", index=False)
    
    accuracy = compute_accuracy(preds, golds)
    print(f"{col} -- Accuracy: {accuracy:.2f}\n")
    with open(f"{prefix}/broca/{model_name}/experiments/{final_csv_subpath}/{col}-accuracy.csv", "w") as f:
        f.write(f"lang,acc\n")
        f.write(f"{col},{accuracy:.2f}\n")

def get_surprisals(logits, ip_tokens, padding_token):
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    target_log_probs =  log_probs[:,:-1].gather(2, ip_tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    target_surprisal = -(target_log_probs / torch.log(torch.tensor(2.0, device=log_probs.device)))
    
    final_list = []
    batch_size = ip_tokens.size(0)
    
    for i in range(batch_size):
        non_padded_tokens = ip_tokens[i, 1:]  # Skip the first token to match logits
        non_padded_surprisals = target_surprisal[i]

        non_padding_indices = (non_padded_tokens != padding_token).nonzero(as_tuple=True)[0]
        non_padding_tokens = non_padded_tokens[non_padding_indices].tolist()
        corresponding_surprisals = non_padded_surprisals[non_padding_indices].tolist()

        token_surprisal_dict = [ (token, surprisal) for token, surprisal in zip(non_padding_tokens, corresponding_surprisals)]
        final_list.append(token_surprisal_dict)
    
    return final_list

def run_standard_experiment(config, col):
    print("Running standard experiment...")
    batch_size = config.get("batch_size")
    final_csv_subpath = config.get("final_csv_subpath")
    prefix = config.get("prefix")
    model_name = config.get("model_name")
    output_path = f"{prefix}/broca/{model_name}/experiments/{final_csv_subpath}/"
    
    tokenizer, model = initialize_model()
    yes_token_id = tokenizer(" Yes")['input_ids']
    no_token_id = tokenizer(" No")['input_ids']

    file_store = pd.DataFrame(columns=["type", "prompt", "q", "prediction", "gold", "surprisal"])
    prompts, questions, golds = get_dataset_from_df(f"{config['prompts_path']}/{col}.csv")
    prompts = [p.strip() for p in prompts]
    preds = []
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i + batch_size]
        batch_golds = golds[i:i + batch_size]
        batch_questions = questions[i:i + batch_size]
        
        tokenizer_inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True)
        batch_predictions = model(**tokenizer_inputs, labels=tokenizer_inputs['input_ids'])
        logits = batch_predictions[1].detach().cpu()     
        yes_answers = logits[:,-1, yes_token_id][:, -1]
        no_answers = logits[:,-1, no_token_id][:, -1]
        answer_ids = torch.where(yes_answers > no_answers, 0, torch.where(yes_answers < no_answers, 1, -1))
        batch_predictions = ["Yes" if idx == 0 else "No" if idx == 1 else "Equal" for idx in answer_ids.tolist()]
        preds = preds + batch_predictions
        
        batch_surprisals = get_surprisals(logits, tokenizer_inputs['input_ids'], tokenizer.eos_token_id)    
        
        for idx in range(len(batch_prompts)):
            batch_file_store = pd.DataFrame([{
                'type': col, 
                'prompt': batch_prompts[idx], 
                'q' :batch_questions[idx], 
                'prediction': batch_predictions[idx], 
                'gold': batch_golds[idx],
                'surprisal': batch_surprisals[idx]
            }])
            file_store = pd.concat([file_store, batch_file_store]).reset_index(drop=True)
    file_store.to_csv(f"{prefix}/broca/{model_name}/experiments/{final_csv_subpath}/{col}.csv", index=False)
    
    accuracy = compute_accuracy(preds, golds)
    print(f"{col} -- Accuracy: {accuracy:.2f}\n")
    with open(f"{prefix}/broca/{model_name}/experiments/{final_csv_subpath}/{col}-accuracy.csv", "w") as f:
        f.write(f"lang,acc\n")
        f.write(f"{col},{accuracy:.2f}\n")

# Main function to run the experiment
def run_experiment(config, args):
    set_environment()
    set_random_seeds()
    config = load_config(args.config)
    
    ablation = config.get("ablation")
    datatype = config.get("datatype")
    # Load data and run the experiment
    data_path = config["data_path"]
    df = pd.read_csv(data_path)
    print(df.columns)
    g_cols = []
    if datatype == "standard":
        g_cols = sorted([col for col in df.columns \
            if not '_S' in col \
            and (not 'qsub' in col) and (not 'null_subject' in col) \
            and ('ita-' in col  or 'en-' in col or 'jap-' in col)
        ])
    elif datatype == "nonce":
        g_cols = [col for col in sorted(df.columns) \
            if not ('ng-' in col) and '_S' in col]
    
    print("Columns ", g_cols)
    col = g_cols[args.stype]
    print(f"Selected column: {col}")
    
    file_path = f'{config["prefix"]}/broca/{config["model_name"]}/experiments/{config["final_csv_subpath"]}/{col}.csv'
    file_path_accuracy = f'{config["prefix"]}/broca/{config["model_name"]}/experiments/{config["final_csv_subpath"]}/{col}-accuracy.csv'
    
    print(file_path)
    if os.path.exists(file_path) and \
       os.path.exists(file_path_accuracy):
           print("Path exists!")
           return
    else:
        # Conditional logic for ablation
        if ablation:
            run_ablation_experiment(config, col)
        else:
            run_standard_experiment(config, col)

# Run if executed as a script
if __name__ == "__main__":
    args = parse_arguments()
    config = load_config(args.config)
    run_experiment(config, args)
