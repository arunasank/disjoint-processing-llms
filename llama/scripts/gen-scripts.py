import argparse
import json
from torch import random, real
from ruamel.yaml import YAML
def set_cuda_visible_devices_and_paths(expt,
                                    available_gpus, 
                                    script_template, 
                                    config_paths_1, 
                                    config_paths_2,
                                    config_paths_3,
                                    expt_output_paths_1, 
                                    expt_output_paths_2,
                                    expt_output_paths_3,
                                    script_path_1,
                                    script_path_2,
                                    run_script_paths):
    num_gpus = len(available_gpus)
    num_files = len(run_script_paths)
    
    # Ensure that the number of config paths and output paths match the number of files
    assert len(config_paths_1) == num_files, "Number of config paths must match the number of run_script_paths."
    assert len(expt_output_paths_1) == num_files, "Number of output paths must match the number of run_script_paths."
    
    # Loop through each file and assign GPUs, config, and output paths
    for idx, run_script_path in enumerate(run_script_paths):
        with open(script_template, 'r') as f:
            content = f.read()
        # Determine which GPU to use for this file
        gpu_id = available_gpus[idx % num_gpus]
        config_path = config_paths_1[idx]
        if expt == 'behav':
            behav_script_path = script_path_1
            expt_output_path = expt_output_paths_1[idx]
            content = content.replace("{OUTPUT_PATH}", expt_output_path)
            content = content.replace("{SCRIPT_PATH}", behav_script_path)    
        elif expt == 'atp-map':
            # rename vars for better readability
            attn_output_path = expt_output_paths_1[idx]
            mlp_output_path = expt_output_paths_2[idx]
            atp_script_path = script_path_1
            map_script_path = script_path_2
            content = content.replace("{ATTN_OUTPUT_PATH}", attn_output_path)
            content = content.replace("{MLP_OUTPUT_PATH}", mlp_output_path)
            content = content.replace("{ATP_SCRIPT_PATH}", atp_script_path)   
            content = content.replace("{MAP_SCRIPT_PATH}", map_script_path)
        elif expt == 'abl':
            real_output_path = expt_output_paths_1[idx]
            unreal_output_path = expt_output_paths_2[idx]
            random_output_path = expt_output_paths_3[idx]
            real_config_path = config_paths_1[idx]
            unreal_config_path = config_paths_2[idx]
            random_config_path = config_paths_3[idx]
            content = content.replace("{SCRIPT_PATH}", script_path_1)
            content = content.replace("{REAL_OUTPUT_PATH}", real_output_path)
            content = content.replace("{UNREAL_OUTPUT_PATH}", unreal_output_path)
            content = content.replace("{RANDOM_OUTPUT_PATH}", random_output_path)
            content = content.replace("{REAL_CONFIG_PATH}", real_config_path)
            content = content.replace("{UNREAL_CONFIG_PATH}", unreal_config_path)
            content = content.replace("{RANDOM_CONFIG_PATH}", random_config_path)
        
        content = content.replace("{CUDA_DEVICE_ID}", gpu_id)
        content = content.replace("{CONFIG_PATH}", config_path)
        
        # Write the new content back to the file
        with open(run_script_path, 'w') as f:
            f.write(content)

def parse_config_file(config_file):
    yaml = YAML()
    with open(config_file, 'rb') as f:
        return yaml.load(f)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set CUDA_VISIBLE_DEVICES and paths for scripts")
    parser.add_argument('--config_file', type=str, required=True, help='Path to the configuration JSON file')
    args = parser.parse_args()
    
    config = parse_config_file(args.config_file)
    
    expt = config['expt']
    avail_gpus = config['gpu_ids']
    script_template = config['script_template']
    run_script_paths = config['run_script_paths']
    
    if expt == 'behav':
        expt_output_paths = config['expt_output_paths']
        script_path = config['script_path']
        config_paths = config['config_paths']
        set_cuda_visible_devices_and_paths(
            expt,
            avail_gpus, 
            script_template, 
            config_paths,
            None,
            None,
            expt_output_paths, 
            None,
            None,
            script_path,
            None,
            run_script_paths
        )
    elif expt == 'atp-map':
        attn_output_paths = config['attn_output_paths']
        mlp_output_paths = config['mlp_output_paths']
        atp_script_path = config['atp_script_path']
        map_script_path = config['map_script_path']
        config_paths = config['config_paths']
        set_cuda_visible_devices_and_paths(
            expt,
            avail_gpus, 
            script_template, 
            config_paths,
            None,
            None,
            attn_output_paths, 
            mlp_output_paths,
            None,
            atp_script_path,
            map_script_path,
            run_script_paths
        )
    elif expt == 'abl':
        real_config_paths = config['real_config_paths']
        unreal_config_paths = config['unreal_config_paths']
        random_config_paths = config['random_config_paths']
        real_output_paths = config['real_output_paths']
        unreal_output_paths = config['unreal_output_paths']
        random_output_paths = config['random_output_paths']
        script_path = config['script_path']
        set_cuda_visible_devices_and_paths(
            expt,
            avail_gpus, 
            script_template, 
            real_config_paths,
            unreal_config_paths,
            random_config_paths,
            real_output_paths,
            unreal_output_paths,
            random_output_paths,
            script_path,
            None,
            run_script_paths
        )
    