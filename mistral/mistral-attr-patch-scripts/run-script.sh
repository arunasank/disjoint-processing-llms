count=30
for i in $(seq $count); do
  > my_script.sh
  echo "#!/bin/bash" >> my_script.sh
  echo "source /etc/profile" >> my_script.sh
  echo "module load anaconda/2023a" >> my_script.sh
  echo "echo $((i-1))" >> my_script.sh
  echo "python3.10 -u mistral-attr-patch.py $((i-1))" >> my_script.sh
  LLsub my_script.sh -s 40 -g volta:1 -o $((i-1)).log
done
