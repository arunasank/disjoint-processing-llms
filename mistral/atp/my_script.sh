#!/bin/bash
source /etc/profile
module load anaconda/2023a
echo 7
python3.10 -u mistral-attr-patch.py --stype 7 --config /home/gridsan/arunas/broca/configs/mistral-atp-config
