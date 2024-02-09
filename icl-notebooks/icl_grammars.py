print('icl_grammars.py')

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig,
    BitsAndBytesConfig
)
import bitsandbytes
from accelerate import infer_auto_device_map
from tqdm import tqdm
from datasets import Dataset, load_dataset
import torch
import random
import pandas as pd 
import sys
import re
COUNT = int(sys.argv[1])

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

MODEL_PATH = '/home/gridsan/arunas/broca/llama/llama-model'
TOKENIZER_PATH = '/home/gridsan/arunas/broca/llama/llama-tokenizer'

# MODEL_PATH = '/home/gridsan/arunas/models/mistralai/Mistral-7B-v0.1/'
# TOKENIZER_PATH = '/home/gridsan/arunas/tokenizers/mistralai/Mistral-7B-v0.1/'

model_path = f"{MODEL_PATH}"
tokenizer_path = f'{TOKENIZER_PATH}'
config = AutoConfig.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_path, config=config, device_map="auto", padding_side="left"
)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(f'{model_path}', device_map="auto", quantization_config=nf4_config)
# model = AutoModelForCausalLM.from_pretrained(f'{model_path}', device_map="auto", load_in_4bit=True)
print('-------------------------Model loaded------------------------------')

device_map = infer_auto_device_map(model)
print('-------------------------Device map--------------------------------')

print(device_map)

print('-------------------------Load dataset-------------------------------')
df = pd.read_csv('ngs.csv')
df = df[ ['sentence', 'subordinate-sentence', 'passive-sentence', 'it', 'it-r-1-null_subject', 'it-r-2-passive', 'it-r-3-subordinate', 'it-u-1-negation', 'it-u-2-invert', 'it-u-3-gender', 'jp-r-1-sov', 'jp-r-2-passive', 'jp-r-3-subordinate', 'jp-u-1-negation',    'jp-u-2-invert', 'jp-u-3-past-tense', 'ng-sentence','ng-subordinate-sentence', 'ng-passive-sentence', 'ng-it','ng-it-r-1-null_subject', 'ng-it-r-2-passive', 'ng-it-r-3-subordinate','ng-it-u-1-negation', 'ng-it-u-2-invert', 'ng-it-u-3-gender','ng-jp-r-1-sov', 'ng-jp-r-2-passive', 'ng-jp-r-3-subordinate','ng-jp-u-1-negation', 'ng-jp-u-2-invert', 'ng-jp-u-3-past-tense']]
gCols = [col for col in df.columns if not 'ng' in col]
print(f"-----------------------Column: {gCols[COUNT:COUNT+1]}----------------------")
def parse_answer(text):
    answer = text.split("A:")[-1].strip()
    return answer

def construct_prompt(train_dataset, num_demonstrations):
    assert num_demonstrations > 0
    prompt = ''
    train_examples = train_dataset.shuffle().select(range(num_demonstrations))
    for exemplar_num in range(num_demonstrations):
        train_example = train_examples[exemplar_num]
        use_bad_sentence = random.choice([True, False])
        exemplar = "Q: Is this sentence grammatical? Yes or No: "
        if use_bad_sentence:
            exemplar += train_example["ng-" + col]
            exemplar += "\nA: No"
        else:
            exemplar += train_example[col]
            exemplar += "\nA: Yes"
        exemplar += "\n\n"
        prompt += exemplar
    return prompt

def compute_accuracy(preds, golds):
    assert len(preds) == len(golds)
    total = 0
    correct = 0
    for pred, gold in zip(preds, golds):
        if pred == gold:
            correct += 1
        total += 1
    return correct / total

@torch.no_grad()
def get_aligned_words_measures(text: str, 
                               measure: str, model, tokenizer) -> list[str]:
    if measure not in {'prob', 'surp'}:
        sys.stderr.write(f"{measure} not recognized\n")
        sys.exit(1)

    data = []

    ids = tokenizer(text, return_tensors='pt').to('cuda')
    input_ids = ids.input_ids.flatten().data
    target_ids = ids.input_ids[:,1:]

    # get output
    logits = model(**ids).logits
    output = torch.nn.functional.log_softmax(logits, dim=-1)
    if measure == 'surp':
        output = -(output/torch.log(torch.tensor(2.0)))
    else:
        output = torch.exp(output)

    # get by token measures 
    target_measures = output[:,:-1, :]
    # use gather to get the output for each target item in the batch
    target_measures = target_measures.gather(-1,
                             target_ids.unsqueeze(2)).flatten().tolist()
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids)[1:]
    words = text.split(' ')
    # A lil loop to force align words 
    current_word = words.pop(0)
    current_token = tokens.pop(0).replace('▁', '')
    measure = 0
    while len(data) != len(text.split(' ')) and len(target_measures) > 0:
        if current_word == current_token:
            data.append((current_word, measure))
            measure = 0
            if words:
                current_word = words.pop(0)
                current_token = tokens.pop(0).replace('▁', '')
                measure += target_measures.pop(0)
        else:
            measure += target_measures.pop(0)
            current_token += tokens.pop(0).replace('▁', '')
            data.append((current_token, measure))

    return data

preds = []
golds = []

f = pd.DataFrame(columns=["type", "prompt", "q", "prediction", "gold", "surprisal", "int-grad"])
f['type'] = 'test'
g = pd.DataFrame(columns=['accuracy', 'type'])

datasets = {}
for col in gCols[COUNT:COUNT+1]:
    datasets[col] = Dataset.from_pandas(pd.DataFrame(df[[col, 'ng-' + col]].copy())).train_test_split(test_size=0.2)

def get_prompt_from_text(filename):
    with open(filename, 'r') as file:
        data = file.read().split('\n-------------------------------------\n')
    
    data = [sentence.strip() for sentence in data]
    data = [sentence for sentence in data if not sentence == '']
    data = [sentence.replace('</s>', '\n') for sentence in data]
    golds = [sentence.strip().split("\n")[-1].strip() for sentence in data]
    data = [sentence[: -len(golds[idx])].strip() for idx, sentence in enumerate(data)]
    data = [sentence[: -len(sentence.strip().split(" ")[-1])].strip() for sentence in data]
    return data, golds 

master_prompt = 'We will provide you a set of sentences which follow or violate a grammatical structure. \n The sentences may use subjects and objects from the following nouns - author, banana, biscuit, book, bottle, box, boy, bulb, cap, cat, chalk, chapter, cucumber, cup, dog, fish, fruit, girl, Gomu, Harry, hill, John, Leela, man, Maria, meal, mountain, mouse, newspaper, pear, pizza, poem, poet, rock, roof, Sheela, speaker, staircase, story, teacher, Tom, toy, tree, woman, writer.\nThe sentences may use any of the following verbs - brings, carries, claims, climbs, eats, holds, notices, reads, says, sees, states, takes.\n Each noun in a sentence may sometimes use a different determiner than those found in English. Here is a reference of determiners that can be used by nouns: "pear": "kar", "author": "kon", "authors": "kons", "banana": "kar", "biscuit": "kon", "book": "kon", "bottle": "kar", "box": "kar", "boy": "kon", "boys": "kons", "bulb": "kar", "cabinet": "kar", "cap": "kon", "cat": "kon", "cats": "kons", "chapter": "kon", "chalk": "kon", "cup": "kar", "cucumber": "kon", "dog": "kon", "dogs": "kons", "fish": "kon", "fruit": "kar", "girl": "kar", "girls": "kars", "hill": "kar", "man": "kon", "men": "kons", "meal": "kon", "mountain": "kar", "mouse": "kon", "newspaper": "kon", "pizza": "kar", "poet": "kon", "poets": "kons", "poem": "kar", "rock": "kon", "roof": "kon", "speaker": "kon", "speakers": "kons", "staircase": "kar", "story": "kar", "teacher": "kon", "teachers": "kons", "toy": "kon", "tree": "kar", "woman": "kar", "women": "kars", "writer": "kon", "writers": "kons". Each verb in a sentence may sometimes use the past tense of the verb if it is more appropriate. Here are a set of verbs and their past tenses - "climbs" : "climbed", "reads": "read", "carries": "carried", "eats": "ate", "holds": "held", "takes" :"took", "brings": "brought", "reads": "read", "climb" : "climbed", "read": "read", "carry": "carried", "eat": "ate", "hold": "held", "take" :"took", "bring": "brought", "read": "read"\n The sentences may sometimes use the infinitive forms of a verb. Here are a set of verbs and their infinitives - "climbs" : "to climb", "reads": "to read", "carries": "to carry", "eats": "to eat", "holds": "to hold", "takes" : "to take", "brings": "to bring", "reads": "to read", "climb" : "to climb", "read": "to read", "carry": "to carry", "eat": "to eat", "hold": "to hold", "take" : "to take", "bring": "to bring", "read": "to read". \n The sentences may sometimes use the plural form of a noun. Here are a set of nouns and their plurals - "fish": "fish", "mouse": "mice", "bottle": "bottles", "newspaper": "newspapers", "chalk": "chalks", "box": "boxes", "cap": "caps", "bulb": "bulbs", "cup": "cups", "toy": "toys", "staircase": "staircases", "rock": "rocks", "hill": "hills", "mountain": "mountains", "roof": "roofs", "tree": "trees", "biscuit": "biscuits", "banana": "bananas", "pear": "pears", "meal": "meals", "fruit": "fruits", "cucumber": "cucumbers", "pizza": "pizzas", "book": "books", "poem": "poems", "story": "stories", "chapter": "chapters". \n The sentences may sometimes use the passive form of a verb. Here are a set of verbs and their passive forms - "carries": "carried", "carry": "carried", "holds": "held", "hold": "held", "takes": "taken", "take": "taken", "brings": "brought", "bring": "brought", "climbs": "climbed", "climb": "climbed", "eats": "eaten", "eat": "eaten", "reads": "read", "read": "read"\n\n'
print('Training.... ')
for NUM_DEMONSTRATIONS in range(10, 15, 5):
    for col in gCols[COUNT:COUNT+1]:
        train_dataset = datasets[col]['train']
        test_dataset = datasets[col]['test']
        prompt = ''
        printAnswer = True
        prompts, golds = get_prompt_from_text(f'classification-train-test-{str(gCols[COUNT])}-prompts.txt')
        # print(prompts[:5], prompts[-5:])
        # print(prompts, golds)
        # for test_sentence in train_dataset:
        for p_idx, prompt in enumerate(prompts):
            try:
                # testBadOrGood = random.choice(['ng-', ''])
                # prompt = construct_prompt(train_dataset, NUM_DEMONSTRATIONS)
                
                # fPrompt = prompt
                
                # # Append test example
                # prompt += "Q: Is this sentence grammatical? Yes or No: "
                # prompt += test_sentence[testBadOrGood + col]
                # prompt += "\nA:"
                
                # fQ = "Q: Is this sentence grammatical? Yes or No: " + test_sentence[testBadOrGood + col] + "\nA:"
                
                # if testBadOrGood == 'ng-':
                #     golds.append("No")
                #     fGold = 'No'
                # else:
                #     golds.append("Yes")
                #     fGold = 'Yes'
                
                # Get answer from model
                model_inputs = tokenizer([master_prompt + prompt], return_tensors="pt").to('cuda')
                # answer = model.generate(prompt_tok,
                #                     top_p=0.9, temperature=0.1,
                #                     max_new_tokens=2)
                answer = model.generate(**model_inputs, pad_token_id=tokenizer.eos_token_id, max_new_tokens=2, top_p=0.9, temperature=0.1, do_sample=True)
                answer = tokenizer.batch_decode(answer)[0]
                if printAnswer:
                    print(answer)
                    printAnswer = False
                preds.append(parse_answer(answer))
                fPrediction = parse_answer(answer)
                # fSurprisal = get_aligned_words_measures(test_sentence[testBadOrGood + col] + " " + parse_answer(answer), "surp", model, tokenizer)
                fSurprisal = get_aligned_words_measures(f'Q: {prompt.split("Q: ")[-1]}' + " " + parse_answer(answer), "surp", model, tokenizer)
                f = pd.concat([f, pd.DataFrame([{'type': col, 'prompt': prompt.replace(f'Q: {prompt.split("Q: ")[-1]}', ''), 'q' : f'Q: {prompt.split("Q: ")[-1]}', 'prediction': fPrediction, 'gold': golds[p_idx], 'surprisal': fSurprisal, 'int-grad': 0}])]).reset_index(drop=True)
                # Evaluate
                # print(" PREDS ####### ", preds, golds)
                # print(" PREDS ####### ", len(preds), len(golds))
            except:
                print(f"###### Exxcept '{prompt}'")
        # print(" PREDS ####### ", len(preds), len(golds))
        accuracy = compute_accuracy(preds, golds)
        print(f"{col} -- Accuracy: {accuracy:.2f}\n")
        g = pd.concat([g, pd.DataFrame([{ 'type' : col, 'accuracy': f"{accuracy:.2f}"}])])


f.to_csv(f"llama-classification-train-test-det-{str(gCols[COUNT])}-new.csv")
g.to_csv(f'llama-classification-train-test-acc-{str(gCols[COUNT])}-new.csv')
