#!/bin/bash
source /etc/profile
module load anaconda/2023a
echo 25
python3.10 -u /home/gridsan/arunas/broca/icl-notebooks/icl_grammars.py 25
