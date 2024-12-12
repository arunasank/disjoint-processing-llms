import yaml
import glob

# DIRS = ['10', '10-seed-1', '10-seed-10', '10-seed-34']
DIRS = ['10-seed-1']
PATH = '/mnt/align4_drive/arunas/broca/configs'
MODELS = ['mistral']

def check_icl(config, model, file, dir):
    assert config['model_name'] == model, f"Model name mismatch in {file}"
    assert config['prefix'] == "/mnt/align4_drive/arunas/", f"Prefix mismatch in {file}"
    assert config['ablation'] == False, f"Ablation mismatch in {file}"
    assert config['data_path'] == "/mnt/align4_drive/arunas/broca/data-gen/prompts/10/full-dataset.csv", f"Data path mismatch in {file}"
    assert config['prompts_path'] == f"/mnt/align4_drive/arunas/broca/data-gen/prompts/{dir}", f"Prompts path mismatch in {file}"
    assert config['num_dems'] == 10, f"Num dems mismatch in {file}"
    assert config['final_csv_subpath'] == f"test-{dir}", f"Final csv subpath mismatch in {file}"
    assert config['datatype'] in ['nonce', 'standard'], f"Type mismatch in {file}"
    if model == 'llama':
        assert len(config.keys()) == 13, f"Number of keys mismatch in {file}"
        assert config['model_path'] == "meta-llama/Llama-2-70b-hf", f"Model path mismatch in {file}"
        assert config['model_cache_path'] == f"/mnt/align4_drive/arunas/{model}-tensors", f"Model cache path mismatch in {file}"
        assert config['batch_size'] == 2, f"Batch size mismatch in {file}"
        assert config['hf_token'] == "${HF_TOKEN}", f"HF token mismatch in {file}"
    elif model == 'mistral':
        assert len(config.keys()) == 10, f"Number of keys mismatch in {file}"
        assert config['model_path'] == "mistralai/Mistral-7B-v0.1", f"Model path mismatch in {file}"
        assert config['batch_size'] == 16, f"Batch size mismatch in {file}"
    else:
        assert False, f"Model name not recognized in {file}"
    
def check_icl_random(config, model, file, dir):
    assert config['model_name'] == model, f"Model name mismatch in {file}"
    assert config['data_path'] == "/mnt/align4_drive/arunas/broca/data-gen/prompts/10/full-dataset.csv", f"Data path mismatch in {file}"
    assert config['prefix'] == "/mnt/align4_drive/arunas/", f"Prefix mismatch in {file}"
    assert config['prompts_path'] == f"/mnt/align4_drive/arunas/broca/data-gen/prompts/{dir}", f"Prompts path mismatch in {file}"
    assert config['patch_pickles_path'] == f"/mnt/align4_drive/arunas/broca/{model}/atp/patches", f"Patch pickles path mismatch in {file}"
    assert config['patch_pickles_subpath'] == f"test-{dir}", f"Patch pickles subpath mismatch in {file}"
    assert config['ablation'] == True, f"Ablation mismatch in {file}"
    assert config['num_dems'] == 10, f"Num dems mismatch in {file}"
    assert config['final_csv_subpath'] == f"test-{dir}-random", f"Final csv subpath mismatch in {file}"
    assert config['hf_token'] == "${HF_TOKEN}", f"HF token mismatch in {file}"
    assert config['topk'] == 0.01, f"Topk mismatch in {file}"
    assert config['randomly_sample'] == True, f"Randomly sample mismatch in {file}"
    assert config['ablate_intersection'] == False, f"Ablate intersection mismatch in {file}"
    assert config['real'] == True, f"Real mismatch in {file}"
    assert config['datatype'] in ['nonce', 'standard'], f"Type mismatch in {file}"
    if model == 'llama':
        assert len(config.keys()) == 19, f"Number of keys mismatch in {file}"
        assert config['model_path'] == "meta-llama/Llama-2-70b-hf", f"Model path mismatch in {file}"
        assert config['model_cache_path'] == f"/mnt/align4_drive/arunas/{model}-tensors", f"Model cache path mismatch in {file}"
        assert config['batch_size'] == 2, f"Batch size mismatch in {file}"
        assert config['hf_token'] == "${HF_TOKEN}", f"HF token mismatch in {file}"
        assert config['num_hidden_states'] == 8192, f"Num hidden states mismatch in {file}"
    elif model == 'mistral':
        assert len(config.keys()) == 18, f"Number of keys mismatch in {file}"
        assert config['model_path'] == "mistralai/Mistral-7B-v0.1", f"Model path mismatch in {file}"
        assert config['batch_size'] == 16, f"Batch size mismatch in {file}"
        assert config['num_hidden_states'] == 4096, f"Num hidden states mismatch in {file}"
    else:
        assert False, f"Model name not recognized in {file}"
        
def check_icl_real(config, model, file, dir):
    assert config['model_name'] == model, f"Model name mismatch in {file}"
    assert config['data_path'] == "/mnt/align4_drive/arunas/broca/data-gen/prompts/10/full-dataset.csv", f"Data path mismatch in {file}"
    assert config['prefix'] == "/mnt/align4_drive/arunas/", f"Prefix mismatch in {file}"
    assert config['prompts_path'] == f"/mnt/align4_drive/arunas/broca/data-gen/prompts/{dir}", f"Prompts path mismatch in {file}"
    assert config['patch_pickles_path'] == f"/mnt/align4_drive/arunas/broca/{model}/atp/patches", f"Patch pickles path mismatch in {file}"
    assert config['patch_pickles_subpath'] == f"test-{dir}", f"Patch pickles subpath mismatch in {file}"
    assert config['ablation'] == True, f"Ablation mismatch in {file}"
    assert config['num_dems'] == 10, f"Num dems mismatch in {file}"
    assert config['final_csv_subpath'] == f"test-{dir}-real", f"Final csv subpath mismatch in {file}"
    assert config['hf_token'] == "${HF_TOKEN}", f"HF token mismatch in {file}"
    assert config['topk'] == 0.01, f"Topk mismatch in {file}"
    assert config['randomly_sample'] == False, f"Randomly sample mismatch in {file}"
    assert config['ablate_intersection'] == True, f"Ablate intersection mismatch in {file}"
    assert config['real'] == True, f"Real mismatch in {file}"
    assert config['datatype'] in ['nonce', 'standard'], f"Type mismatch in {file}"
    if model == 'llama':
        assert len(config.keys()) == 19, f"Number of keys mismatch in {file}"
        assert config['model_path'] == "meta-llama/Llama-2-70b-hf", f"Model path mismatch in {file}"
        assert config['model_cache_path'] == f"/mnt/align4_drive/arunas/{model}-tensors", f"Model cache path mismatch in {file}"
        assert config['batch_size'] == 2, f"Batch size mismatch in {file}"
        assert config['hf_token'] == "${HF_TOKEN}", f"HF token mismatch in {file}"
        assert config['num_hidden_states'] == 8192, f"Num hidden states mismatch in {file}"
    elif model == 'mistral':
        assert len(config.keys()) == 18, f"Number of keys mismatch in {file}"
        assert config['model_path'] == "mistralai/Mistral-7B-v0.1", f"Model path mismatch in {file}"
        assert config['batch_size'] == 16, f"Batch size mismatch in {file}"
        assert config['num_hidden_states'] == 4096, f"Num hidden states mismatch in {file}"
    else:
        assert False, f"Model name not recognized in {file}"

def check_icl_unreal(config, model, file, dir):
    assert config['model_name'] == model, f"Model name mismatch in {file}"
    assert config['data_path'] == "/mnt/align4_drive/arunas/broca/data-gen/prompts/10/full-dataset.csv", f"Data path mismatch in {file}"
    assert config['prefix'] == "/mnt/align4_drive/arunas/", f"Prefix mismatch in {file}"
    assert config['prompts_path'] == f"/mnt/align4_drive/arunas/broca/data-gen/prompts/{dir}", f"Prompts path mismatch in {file}"
    assert config['patch_pickles_path'] == f"/mnt/align4_drive/arunas/broca/{model}/atp/patches", f"Patch pickles path mismatch in {file}"
    assert config['patch_pickles_subpath'] == f"test-{dir}", f"Patch pickles subpath mismatch in {file}"
    assert config['ablation'] == True, f"Ablation mismatch in {file}"
    assert config['num_dems'] == 10, f"Num dems mismatch in {file}"
    assert config['final_csv_subpath'] == f"test-{dir}-unreal", f"Final csv subpath mismatch in {file}"
    assert config['hf_token'] == "${HF_TOKEN}", f"HF token mismatch in {file}"
    assert config['topk'] == 0.01, f"Topk mismatch in {file}"
    assert config['randomly_sample'] == False, f"Randomly sample mismatch in {file}"
    assert config['ablate_intersection'] == True, f"Ablate intersection mismatch in {file}"
    assert config['real'] == False, f"Real mismatch in {file}"
    assert config['datatype'] in ['nonce', 'standard'], f"Type mismatch in {file}"
    if model == 'llama':
        assert len(config.keys()) == 19, f"Number of keys mismatch in {file}"
        assert config['model_path'] == "meta-llama/Llama-2-70b-hf", f"Model path mismatch in {file}"
        assert config['model_cache_path'] == f"/mnt/align4_drive/arunas/{model}-tensors", f"Model cache path mismatch in {file}"
        assert config['batch_size'] == 2, f"Batch size mismatch in {file}"
        assert config['hf_token'] == "${HF_TOKEN}", f"HF token mismatch in {file}"
        assert config['num_hidden_states'] == 8192, f"Num hidden states mismatch in {file}"
    elif model == 'mistral':
        assert len(config.keys()) == 18, f"Number of keys mismatch in {file}"
        assert config['model_path'] == "mistralai/Mistral-7B-v0.1", f"Model path mismatch in {file}"
        assert config['batch_size'] == 16, f"Batch size mismatch in {file}"
        assert config['num_hidden_states'] == 4096, f"Num hidden states mismatch in {file}"
    else:
        assert False, f"Model name not recognized in {file}"

def check_atp(config, model, file, dir):
    assert config["model_name"] == model, f"Model name mismatch in {file}"
    assert config["data_path"] == "/mnt/align4_drive/arunas/broca/data-gen/prompts/10/full-dataset.csv", f"Data path mismatch in {file}"
    assert config["prefix"] == "/mnt/align4_drive/arunas/", f"Prefix mismatch in {file}"
    assert config["prompt_files_path"] == f"/mnt/align4_drive/arunas/broca/data-gen/prompts/{dir}", f"Prompts path mismatch in {file}"
    assert config["patch_pickles_path"] == f"/mnt/align4_drive/arunas/broca/{model}/atp/patches", f"Patch pickles path mismatch in {file}"
    assert config["patch_pickles_sub_path"] == f"test-{dir}", f"Patch pickles subpath mismatch in {file}"
    assert config['datatype'] in ['nonce', 'standard'], f"Type mismatch in {file}"
    if model == 'llama':
        assert len(config.keys()) == 10, f"Number of keys mismatch in {file}"
        assert config['model_path'] == "meta-llama/Llama-2-70b-hf", f"Model path mismatch in {file}"
        assert config['model_cache_path'] == f"/mnt/align4_drive/arunas/{model}-tensors", f"Model cache path mismatch in {file}"
        assert config['hf_token'] == "${HF_TOKEN}", f"HF token mismatch in {file}"
    elif model == 'mistral':
        assert len(config.keys()) == 8, f"Number of keys mismatch in {file}"
        assert config['model_path'] == "mistralai/Mistral-7B-v0.1", f"Model path mismatch in {file}"
    else:
        assert False, f"Model name not recognized in {file}"
        
def check_map(config, model, file, dir):
    assert config['model_name'] == model, f"Model name mismatch in {file}"
    assert config['prompt_files_path'] == f"/mnt/align4_drive/arunas/broca/data-gen/prompts/{dir}", f"Prompt files path mismatch in {file}"
    assert config['patch_pickles_path'] == f"/mnt/align4_drive/arunas/broca/{model}/atp/patches", f"Patch pickles path mismatch in {file}"
    assert config['patch_pickles_sub_path'] == f"test-{dir}", f"Patch pickles subpath mismatch in {file}"
    assert config['prefix'] == "/mnt/align4_drive/arunas/", f"Prefix mismatch in {file}"
    assert config['data_path'] == "/mnt/align4_drive/arunas/broca/data-gen/prompts/10/full-dataset.csv", f"Data path mismatch in {file}"
    assert config['datatype'] in ['nonce', 'standard'], f"Type mismatch in {file}"
    if model == 'llama':
        assert len(config.keys()) == 10, f"Number of keys mismatch in {file}"
        assert config['model_path'] == "meta-llama/Llama-2-70b-hf", f"Model path mismatch in {file}"
        assert config['model_cache_path'] == f"/mnt/align4_drive/arunas/{model}-tensors", f"Model cache path mismatch in {file}"
        assert config['hf_token'] == "${HF_TOKEN}", f"HF token mismatch in {file}"
    elif model == 'mistral':
        assert len(config.keys()) == 8, f"Number of keys mismatch in {file}"
        assert config['model_path'] == "mistralai/Mistral-7B-v0.1", f"Model path mismatch in {file}"
    
def run_checks(DIRS, MODELS, PATH):
    for dir in DIRS:
        for model in MODELS:
            for file in glob.glob(f'{PATH}/{dir}/{model}*'):
                with open( file, 'r') as f:
                    config = yaml.safe_load(f)
                    print(f"Checking {file}")
                    if (f'{model}-icl-config' in file[-len(f'{model}-icl-config'):]):
                        check_icl(config, model, file, dir)
                    elif (f'{model}-icl-config-random' in file[-len(f'{model}-icl-config-random'):]):
                        check_icl_random(config, model, file, dir)
                    elif (f'{model}-icl-config-real' in file[-len(f'{model}-icl-config-real'):]):
                        check_icl_real(config, model, file, dir)
                    elif (f'{model}-icl-config-unreal' in file[-len(f'{model}-icl-config-unreal'):]):
                        check_icl_unreal(config, model, file, dir)
                    elif (f'{model}-atp-config' in file[-len(f'{model}-atp-config'):]):
                        check_atp(config, model, file, dir)
                    elif (f'{model}-map-config' in file[-len(f'{model}-map-config'):]):
                        check_map(config, model, file, dir)
                    else:
                        assert False, f"File name not recognized in {file}"
                    
if __name__ == "__main__":
    run_checks(DIRS, MODELS, PATH)