#!/bin/bash
# Loading the required module
source /etc/profile
module load anaconda/2023a
#nvidia-smi
#pip install transformers
#pip install bitsandbytes
#pip install torch datasets accelerate
#pip install random
#pip install pandas
#pip install torch
# Run the script
#python3.10 -u llama-attr-patching.py
python3.10 -u icl_grammars.py 0
