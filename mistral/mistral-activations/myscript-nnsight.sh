#!/bin/bash
source /etc/profile
module load anaconda/2023a
echo 15
python3.10 -u nnsight-my-file.py
