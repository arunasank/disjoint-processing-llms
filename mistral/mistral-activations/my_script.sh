#!/bin/bash
source /etc/profile
module load anaconda/2023a
echo 15
python3.10 -u l-script.py 15
