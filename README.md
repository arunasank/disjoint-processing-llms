# Disjoint Processing Mechanisms of Hierarchical and Linear Grammars in Large Language Models

> All natural languages are structured hierarchically. In humans, this structural restriction is neurologically coded: when two grammars are presented with identical vocabularies, brain areas responsible for language processing are only sensitive to hierarchical grammars. Using large language models (LLMs), we investigate whether such functionally distinct hierarchical processing regions can arise solely from exposure to large-scale language distributions. We generate inputs using English, Italian, Japanese, or nonce words, varying the underlying grammars to conform to either hierarchical or linear/positional rules. Using these grammars, we first observe that language models show distinct behaviors on hierarchical versus linearly structured inputs. Then, we find that the components responsible for processing hierarchical grammars are distinct from those that process linear grammars; we causally verify this in ablation experiments. Finally, we observe that hierarchy-selective components are also active on nonce grammars; this suggests that hierarchy sensitivity is not tied to meaning, nor in-distribution inputs.

## What is this repository?
This repository contains datasets (./data) and code (./exp-1-4-5 and ./exp-2-3-5) to replicate the experiments in our paper.

### Experiment 1
Experiment 1 compares the performance of a pre-trained LLM on grammaticality judgment
tasks given hierarchical and linear grammars
To run experiment 1:
* Generate the config file for experiment 1 using the `utils/gen-config.py` file: `python utils/gen-config.py --which beh --tok conventional` or `python utils/gen-config.py --which beh --tok nonce` depending on whether you are replicating the experiment with the conventional or jabberwocky sentences. 
* Call `python ./exp-1-4-5/behavioural.py --config <path to the generated config file> --stype <grammar structure index> --batch_size`

## Experiment 2
Experiment 2 locates model components that are important for processing hierarchical and linear structures by treating hierarchical and linear inputs as counterfactuals
* Generate the config file for experiment 2 using the `utils/gen-config.py` file: `python utils/gen-config.py --which atp --tok conventional` or `python utils/gen-config.py --which atp --tok nonce`
* Call `python ./exp-2-3-5/atp.py --config <path to the generated config file> --stype <grammar structure index>`

## Experiment 3
Experiment 3 investigates the causal role of these components by ablating them and then measuring changes in grammaticality judgment performance
* Generate the config file for experiment 3 using the `utils/gen-config.py` file:
  * Ablate random components: `python utils/gen-config.py --which abl-ra --tok conventional` or `python utils/gen-config.py --which abl-ra --tok nonce`
  * Ablate components sensitive to real grammars: `python utils/gen-config.py --which abl-r --tok conventional` or `python utils/gen-config.py --which abl-r --tok nonce`
  * Ablate components sensitive to unreal grammars: `python utils/gen-config.py --which abl-u --tok conventional` or `python utils/gen-config.py --which abl-u --tok nonce`
* Call `python ./exp-1-4-5/behavioural.py --config <path to the generated config file> --stype <grammar structure index> --batch_size`. Depending on the config files provided, components will be ablated.

## Experiment 4
Experiment 4 investigates whether the components identified in Experiment 2 merely   distinguish grammars that are in-distribution with the training data, or show a more abstract universal sensitivity to hierarchical and linear structure on nonce sentences. For this experiment, we ablate components sensitive to the conventional hierarchical and linear structures when testing the model's performance on the nonce structures. Generate config files accordingly for this task, based on instructions for experiments 1, 2, and 3.




