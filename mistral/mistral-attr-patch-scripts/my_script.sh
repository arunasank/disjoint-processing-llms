#!/bin/bash
source /etc/profile
module load anaconda/2023a
echo 30
python3.10 -u mistral-attr-patch.py 30
