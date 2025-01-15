import os
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--which', type=str, help='config file for which expt (beh, atp, map, abl-r, abl-u, abl-ra)')
parser.add_argument('--tok', type=str, default='conventional', help='which token type - conventional or nonce')

args = parser.parse_args()

# Define models and their paths
models = {
    "llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "mistral-v0.3": "mistralai/Mistral-7B-v0.3",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
    "gpt-2-xl": "openai-community/gpt2-xl",
    "qwen-2-0.5b": "Qwen/Qwen2-0.5B",
    "llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "qwen-2-1.5b": "Qwen/Qwen2-1.5B",
    "llama-3.1-70b": "meta-llama/Llama-3.1-70B"
}
which = args.which
tok = args.tok
sub_path = f"{tok}-10"
data_path = "/mnt/align4_drive/arunas/broca/data-gen/prompts/10/full-dataset.csv" if tok == "conventional" else "/mnt/align4_drive/arunas/broca/data-gen/ngs-08-01-2024-synthetic-grammars-nonce.csv"
prompts_path = "/mnt/align4_drive/arunas/broca/data-gen/prompts/10-seed-10" if tok == "conventional" else "/mnt/align4_drive/arunas/broca/data-gen/prompts/synthetic/10"

template = None
# Define file content map_template
if which == "map":
    template = """model_name: "{model_name}"
model_path: "{model_path}"
prefix: "/mnt/align4_drive/arunas/"
data_path: "{data_path}"
prompt_files_path: "{prompts_path}"
patch_pickles_path: "/mnt/align4_drive/arunas/broca/{model_name}/atp/patches"
patch_pickles_sub_path: "{subpath}"
datatype: "{tok}"
"""
elif which == "beh":
    template = """model_name: "{model_name}"
model_path: "{model_path}"
prefix: "/mnt/align4_drive/arunas/"
ablation: False
data_path: "{data_path}"
prompts_path: "{prompts_path}"
num_dems: 10
final_csv_subpath: "{subpath}"
hf_token: ${{HF_TOKEN}}
topk: 0.01
datatype: "{tok}"
"""
elif which == "atp":
    template = """model_name: "{model_name}"
model_path: "{model_path}"
prefix: "/mnt/align4_drive/arunas/"
data_path: "{data_path}"
prompt_files_path: "{prompts_path}"
patch_pickles_path: "/mnt/align4_drive/arunas/broca/{model_name}/atp/patches"
patch_pickles_sub_path: "{subpath}"
datatype: "{tok}"
"""
elif which == "abl-r":
    template = """model_name: "{model_name}"
model_path: "{model_path}"
prefix: "/mnt/align4_drive/arunas/"
data_path: "{data_path}"
prompts_path: "{prompts_path}"
patch_pickles_path: "/mnt/align4_drive/arunas/broca/{model_name}/atp/patches"
patch_pickles_subpath: "{subpath}"
ablation: True
ablation_type: "grammar-specific"
token_pos_type: "all"
num_dems: 10
final_csv_subpath: "{subpath}-real-grammar-specific-all"
hf_token: ${{HF_TOKEN}}
topk: 0.01
random_ablate: False
intersection_ablate: False
real: True
datatype: "{tok}"
"""
elif which == "abl-u":
    template = """model_name: "{model_name}"
model_path: "{model_path}"
data_path: "{data_path}"
prefix: "/mnt/align4_drive/arunas/"
prompts_path: "{prompts_path}"
patch_pickles_path: "/mnt/align4_drive/arunas/broca/{model_name}/atp/patches"
patch_pickles_subpath: "{subpath}"
ablation: True
ablation_type: "grammar-specific"
token_pos_type: "all"
num_dems: 10
final_csv_subpath: "{subpath}-unreal-grammar-specific-all"
hf_token: ${{HF_TOKEN}}
topk: 0.01
random_ablate: False
intersection_ablate: False
real: False
datatype: "{tok}"
"""
elif which == "abl-ra":
    template = """model_name: "{model_name}"
model_path: "{model_path}"
data_path: "{data_path}"
prompts_path: "{prompts_path}"
prefix: "/mnt/align4_drive/arunas/"
patch_pickles_path: "/mnt/align4_drive/arunas/broca/{model_name}/atp/patches"
patch_pickles_subpath: "{subpath}"
ablation: True
ablation_type: "grammar-specific"
token_pos_type: "all"
num_dems: 10
final_csv_subpath: "{subpath}-random-grammar-specific-all"
hf_token: ${{HF_TOKEN}}
topk: 0.01
random_ablate: True
intersection_ablate: False
real: True
datatype: "{tok}"
"""

# Output directory
output_dir = f"/mnt/align4_drive/arunas/broca/configs/{sub_path}/"
os.makedirs(output_dir, exist_ok=True)

assert template is not None, "Template not defined."
# Generate files
for model_name, model_path in models.items():
    file_content = template.format(model_name=model_name, model_path=model_path, data_path=data_path, prompts_path=prompts_path, subpath=sub_path, tok=tok)
    file_name = f"{model_name}-{which}-config"
    file_path = os.path.join(output_dir, file_name)
    
    # Write the file
    with open(file_path, "w") as f:
        f.write(file_content)
        print(f"Generated {file_path}")

print("All configuration files generated.")
