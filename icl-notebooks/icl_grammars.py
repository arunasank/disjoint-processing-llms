from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig,
    BitsAndBytesConfig
)
import bitsandbytes
from accelerate import infer_auto_device_map

from datasets import Dataset, load_dataset
import torch
import random
import pandas as pd
import os

os.environ['HF_TOKEN'] = "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"
PREFIX = '/mnt/align4_drive/arunas'

nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                    )

config = AutoConfig.from_pretrained("meta-llama/Llama-2-70b-hf")
tokenizer = AutoTokenizer.from_pretrained(
           "meta-llama/Llama-2-70b-hf", config=config, device_map="auto", padding_side="left"
           )

tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b-hf",  quantization_config=nf4_config, device_map='auto', cache_dir='/mnt/align4_drive/arunas/llama-tensors/') # Load the model

device_map = infer_auto_device_map(model)
print(device_map)
df = pd.read_csv(f'{PREFIX}/broca/data-gen/ngs.csv')
cols = list(df.columns)

def parse_answer(text):
    answers = []
    for t in text:
        ans = t.split("A:")[-1].strip()
        answers.append(ans)
    return answers

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

import sys
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

@torch.no_grad()
def get_aligned_words_measures(texts: str, 
                               answer: str,
                               measure: str,
                               model: GPT2LMHeadModel, 
                               tokenizer: GPT2Tokenizer) -> list[str]:
    """ Returns words and their measure (prob|surp)
    Args:
        text (list[str]): list of sentences
        measure (str): Measure, either probability or surprisal
                        (options: prob|surp)
        model (GPT2LMHeadModel): Pretrained model
        tokenizer (GPT2Tokenizer): Tokenizer
    Returns:
        list[str]: List of words with their measures

    For example, 
    >>> model, tokenizer = load_pretrained_model()
    >>> get_aligned_words_measures('the student is happy', 
    ...        'surp', model, tokenizer)
    [('the', 0), ('student', 17.38616943359375), ('is', 6.385905742645264),
     ('happy', 9.564245223999023)]
    >>> get_aligned_words_measures('the cat is fluffy', 
    ...        'prob', model, tokenizer) 
    [('the', 0), ('cat', 2.5601848392398097e-06), ('is', 0.025296149775385857),
     ('fluffy', 0.00020585735910572112)]
    >>> get_aligned_words_measures('the cat are fluffy', 
    ...        'prob', model, tokenizer)
    [('the', 0), ('cat', 2.5601848392398097e-06), ('are', 0.0010310395155102015),
     ('fluffy', 0.00021902224398218095)]
    """
    if measure not in {'prob', 'surp'}:
        sys.stderr.write(f"{measure} not recognized\n")
        sys.exit(1)

    datas = []
    for t in range(len(texts)):
        text = f'{texts[t]} {answers[t]}'
        data = []
    
        ids = tokenizer(text, return_tensors='pt')
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
        datas.append(data)
    return datas

preds = []
golds = []

f = pd.DataFrame(columns=["type", "prompt", "q", "prediction", "gold", "surprisal", "int-grad"])
f['type'] = 'test'
g = pd.DataFrame(columns=['accuracy', 'type'])

gCols = [col for col in list(df.columns) if not 'ng' in col]
datasets = {}
for col in gCols:
    datasets[col] = Dataset.from_pandas(pd.DataFrame(df[[col, 'ng-' + col]].copy())).train_test_split(test_size=0.2)

master_prompt = 'We will provide you a set of sentences which follow or violate a grammatical structure. \n The sentences may use subjects and objects from the following nouns - author, banana, biscuit, book, bottle, box, boy, bulb, cap, cat, chalk, chapter, cucumber, cup, dog, fish, fruit, girl, Gomu, Harry, hill, John, Leela, man, Maria, meal, mountain, mouse, newspaper, pear, pizza, poem, poet, rock, roof, Sheela, speaker, staircase, story, teacher, Tom, toy, tree, woman, writer.\nThe sentences may use any of the following verbs - brings, carries, claims, climbs, eats, holds, notices, reads, says, sees, states, takes.\n Each noun in a sentence may sometimes use a different determiner than those found in English. Here is a reference of determiners that can be used by nouns: "pear": "kar", "author": "kon", "authors": "kons", "banana": "kar", "biscuit": "kon", "book": "kon", "bottle": "kar", "box": "kar", "boy": "kon", "boys": "kons", "bulb": "kar", "cabinet": "kar", "cap": "kon", "cat": "kon", "cats": "kons", "chapter": "kon", "chalk": "kon", "cup": "kar", "cucumber": "kon", "dog": "kon", "dogs": "kons", "fish": "kon", "fruit": "kar", "girl": "kar", "girls": "kars", "hill": "kar", "man": "kon", "men": "kons", "meal": "kon", "mountain": "kar", "mouse": "kon", "newspaper": "kon", "pizza": "kar", "poet": "kon", "poets": "kons", "poem": "kar", "rock": "kon", "roof": "kon", "speaker": "kon", "speakers": "kons", "staircase": "kar", "story": "kar", "teacher": "kon", "teachers": "kons", "toy": "kon", "tree": "kar", "woman": "kar", "women": "kars", "writer": "kon", "writers": "kons". Each verb in a sentence may sometimes use the past tense of the verb if it is more appropriate. Here are a set of verbs and their past tenses - "climbs" : "climbed", "reads": "read", "carries": "carried", "eats": "ate", "holds": "held", "takes" :"took", "brings": "brought", "reads": "read", "climb" : "climbed", "read": "read", "carry": "carried", "eat": "ate", "hold": "held", "take" :"took", "bring": "brought", "read": "read"\n The sentences may sometimes use the infinitive forms of a verb. Here are a set of verbs and their infinitives - "climbs" : "to climb", "reads": "to read", "carries": "to carry", "eats": "to eat", "holds": "to hold", "takes" : "to take", "brings": "to bring", "reads": "to read", "climb" : "to climb", "read": "to read", "carry": "to carry", "eat": "to eat", "hold": "to hold", "take" : "to take", "bring": "to bring", "read": "to read". \n The sentences may sometimes use the plural form of a noun. Here are a set of nouns and their plurals - "fish": "fish", "mouse": "mice", "bottle": "bottles", "newspaper": "newspapers", "chalk": "chalks", "box": "boxes", "cap": "caps", "bulb": "bulbs", "cup": "cups", "toy": "toys", "staircase": "staircases", "rock": "rocks", "hill": "hills", "mountain": "mountains", "roof": "roofs", "tree": "trees", "biscuit": "biscuits", "banana": "bananas", "pear": "pears", "meal": "meals", "fruit": "fruits", "cucumber": "cucumbers", "pizza": "pizzas", "book": "books", "poem": "poems", "story": "stories", "chapter": "chapters". \n The sentences may sometimes use the passive form of a verb. Here are a set of verbs and their passive forms - "carries": "carried", "carry": "carried", "holds": "held", "hold": "held", "takes": "taken", "take": "taken", "brings": "brought", "bring": "brought", "climbs": "climbed", "climb": "climbed", "eats": "eaten", "eat": "eaten", "reads": "read", "read": "read"\n\n'
NUM_DEMONSTRATIONS = 10
BATCH_SIZE = 64
for col in gCols:
    train_dataset = datasets[col]['train']
    test_dataset = datasets[col]['test']
    printAnswer = False
    
    for i in range(0, len(test_dataset), BATCH_SIZE):
        test_sentences = []
        fPrompts = []
        fQs = []
        fGolds = []
        prompts = []
        for batch_idx in range(BATCH_SIZE):
            #print(f'batch_idx: {i + batch_idx} {len(test_dataset)}')
            testBadOrGood = random.choice(['ng-', ''])
            if (i + batch_idx) >= len(test_dataset):
                break;
            test_sentence = test_dataset[i + batch_idx]
            prompt = construct_prompt(train_dataset, NUM_DEMONSTRATIONS)
            
            fPrompt = prompt
            
            # Append test example
            prompt += "Q: Is this sentence grammatical? Yes or No: "
            prompt += test_sentence[testBadOrGood + col]
            prompt += "\nA:"
            
            fQ = "Q: Is this sentence grammatical? Yes or No: " + test_sentence[testBadOrGood + col] + "\nA:"
            
            if testBadOrGood == 'ng-':
                golds.append("No")
                fGold = 'No'
            else:
                golds.append("Yes")
                fGold = 'Yes'
            fGolds.append(fGold)
            prompts.append(master_prompt + prompt)
            test_sentences.append(test_sentence[testBadOrGood + col])
            fPrompts.append(fPrompt)
            fQs.append(fQ)
            
        # Get answer from model
        model_inputs = tokenizer(prompts, return_tensors="pt", padding=True).to('cuda')
        answers = model.generate(**model_inputs, pad_token_id=tokenizer.eos_token_id, max_new_tokens=2, top_p=0.9, temperature=0.1, do_sample=True)
        answers = tokenizer.batch_decode(answers)[:BATCH_SIZE]

        if printAnswer:
            print(answers)
            printAnswer = False
        
        preds = preds + parse_answer(answers)
        fPredictions = parse_answer(answers)
        fSurprisals = get_aligned_words_measures(test_sentences, parse_answer(answers), "surp", model, tokenizer)
        for batch_idx in range(len(fPrompts)):
            f = pd.concat([f, pd.DataFrame([{'type': col, 'prompt': fPrompts[batch_idx], 'q' :fQs[batch_idx], 'prediction': fPredictions[batch_idx], 'gold': fGolds[batch_idx], 'surprisal': fSurprisals[batch_idx], 'int-grad': 0}])]).reset_index(drop=True)
    # Evaluate
    accuracy = compute_accuracy(preds, golds)
    print(f"{col} -- Accuracy: {accuracy:.2f}\n")
    g = pd.concat([g, pd.DataFrame([{ 'trainType' : col, 'testType': col, 'accuracy': f"{accuracy:.2f}"}])])
    f.to_csv(f"{PREFIX}/broca/mistral/experiments/mistral-classification-train-test-det-{col}-new-1.csv")

g.to_csv(f'{PREFIX}/broca/mistral/experiments/mistral-classification-train-test-acc-new-1.csv')
