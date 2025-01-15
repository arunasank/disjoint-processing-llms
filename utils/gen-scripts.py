import argparse
import json
from torch import random, real
from ruamel.yaml import YAML

def parse_config_file(config_file):
    yaml = YAML()
    with open(config_file, 'rb') as f:
        return yaml.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set CUDA_VISIBLE_DEVICES and paths for scripts")
    parser.add_argument('--config_file', type=str, required=True, help='Path to the configuration JSON file')
    args = parser.parse_args()
    
    config = parse_config_file(args.config_file)
    
    num_seeds = len(config['beh_config_paths'])
    
    for idx in range(num_seeds):
        atp_attn_output_path = config['atp_attn_output_paths']
        atp_config_path = config['atp_config_paths']
        atp_mlp_output_path = config['atp_mlp_output_paths']
        atp_script_path = config['atp_script_path']
        beh_config_path = config['beh_config_paths']
        beh_output_path = config['beh_output_paths']
        beh_random_config_path = config['beh_random_config_paths']
        beh_random_output_path = config['beh_random_output_paths']
        beh_real_config_path = config['beh_real_config_paths']
        beh_real_output_path = config['beh_real_output_paths']
        beh_script_path = config['beh_script_path']
        beh_unreal_config_path = config['beh_unreal_config_paths']
        beh_unreal_output_path = config['beh_unreal_output_paths']
        cuda_device_id = config['cuda_device_ids']
        map_script_path = config['map_script_path']
        prefix = config['op_run_script_prefix']
        for tidx, template in enumerate(config['script_templates']):
            with open(template, 'r') as f:
                script = f.read()
            script = script.replace('{ATP_SCRIPT_PATH}', atp_script_path)
            script = script.replace('{BEH_SCRIPT_PATH}', beh_script_path)
            script = script.replace('{MAP_SCRIPT_PATH}', map_script_path)
            
            script = script.replace('{ATP_ATTN_OUTPUT_PATH}', atp_attn_output_path[idx])
            script = script.replace('{ATP_CONFIG_PATH}', atp_config_path[idx])
            script = script.replace('{ATP_MLP_OUTPUT_PATH}', atp_mlp_output_path[idx])
            script = script.replace('{BEH_CONFIG_PATH}', beh_config_path[idx])
            script = script.replace('{BEH_OUTPUT_PATH}', beh_output_path[idx])
            script = script.replace('{BEH_RANDOM_CONFIG_PATH}', beh_random_config_path[idx])
            script = script.replace('{BEH_RANDOM_OUTPUT_PATH}', beh_random_output_path[idx])
            script = script.replace('{BEH_REAL_CONFIG_PATH}', beh_real_config_path[idx])
            script = script.replace('{BEH_REAL_OUTPUT_PATH}', beh_real_output_path[idx])
            script = script.replace('{BEH_UNREAL_CONFIG_PATH}', beh_unreal_config_path[idx])
            script = script.replace('{BEH_UNREAL_OUTPUT_PATH}', beh_unreal_output_path[idx])
            script = script.replace('{CUDA_DEVICE_ID}', str(cuda_device_id[tidx]))
            with open(f'{prefix}-{idx}-{tidx}', 'w') as f:
                f.write(script)
    