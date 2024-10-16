#!/bin/bash

set -x

# Function to run a script on a GPU and check its status
run_script() {
  local script=$1
  bash "$script"
  if [ $? -ne 0 ]; then
    echo "Script $script failed. Aborting."
    exit 1
  fi
}

# Function to run scripts in parallel and wait for all to complete
run_scripts_in_parallel() {
  local scripts=("$@")
  local pids=()
  
  for script in "${scripts[@]}"; do
    run_script $script &
    pids+=($!)
  done

  # Wait for all scripts to finish
  for pid in "${pids[@]}"; do
    wait $pid
    if [ $? -ne 0 ]; then
      echo "One of the scripts failed. Aborting."
      exit 1
    fi
  done
}

echo "######### Llama experiment 1"
run_scripts_in_parallel expt1/lrig.aa.sh expt1/lrig.ab.sh expt1/lrig.ac.sh expt1/lrig.ad.sh

#run_scripts_in_parallel expt1/lrig.ad

# echo "######### Llama experiment 1, jap-r-2-subordinate"
#run_scripts_in_parallel expt1/13

echo "######### Llama experiment 3, ATP"
run_scripts_in_parallel expt3/rlig.aa.sh rlig.ab.sh rlig.ac.sh rlig.ad.sh 

echo "######### Llama experiment 3, ATP, jap-r-2-subordinate"
#run_scripts_in_parallel expt3/13-rlig

#echo "######### Llama experiment 3, MAP"
run_scripts_in_parallel expt3/mean-rlig.aa.sh expt3/mean-rlig.ab.sh expt3/mean-rlig.ac.sh expt3/mean-rlig.ad.sh
#run_scripts_in_parallel expt3/mean-rlig.ad
  
#echo "######### Llama experiment 3, 13 MAP"
#run_scripts_in_parallel expt3/13

echo "######### Llama experiment 4, ABLATIONS"
run_scripts_in_parallel expt4/lrig.aa.sh expt4/lrig.ab.sh expt4/lrig.ac.sh expt4/lrig.ad.sh

run_scripts_in_parallel expt4/lrig.ad expt4/lrig.ad1 expt4/lrig.ad2 

# echo "######### Llama experiment 4, 13 ABLATIONS"
# run_scripts_in_parallel expt4/13 expt4/13.1 expt4/13.2

echo "########## SUCCESS!!! All scripts have completed successfully."
