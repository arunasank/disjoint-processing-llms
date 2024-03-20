#!/bin/bash
source /etc/profile
module load anaconda/2023a
echo 0
python3.10 -u mistral-attr-patch.py 0
