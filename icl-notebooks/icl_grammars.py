from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig,
    BitsAndBytesConfig
)
import bitsandbytes
from accelerate import infer_auto_device_map

import sys
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import Dataset, load_dataset
import torch
import random
import pandas as pd
import os
import pickle
import yaml
import argparse

random.seed(42)
torch.manual_seed(42)

parser = argparse.ArgumentParser()

parser.add_argument('--config', type=str, help='path to the model training config file, found in broca/configs')

args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)


PREFIX = config.prefix
MODEL_NAME = config.model_name
MODEL_PATH = config.model_path
ABLATION = config.ablation
DATA_PATH = config.data_path
NUM_DEMONSTRATIONS = config.num_demonstrations
BATCH_SIZE = config.batch_size
FINAL_CSV_PATH = config.final_csv_path

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_config = AutoConfig.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, config=model_config, device_map="auto", padding_side="left")

tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, config=model_config, quantization_config=nf4_config, device_map='auto') # Load the model

device_map = infer_auto_device_map(model)

df = pd.read_csv(f'{DATA_PATH}')
gCols = [col for col in list(df.columns) if not 'ng' in col]
col = gCols[int(sys.argv[1])]

if (ABLATION):
    MEAN_ABLATION = config.mean_ablation
    ABLATION_PICKLE_PREFIX = config.ablation_prefix
    ABLATION_PICKLE_SUFFIX = config.ablation_suffix
    def ablate_model(col):
        with open(f'{PREFIX}/broca/{MODEL_NAME}/{MODEL_NAME}-attr-patch-scripts/mlp/{ABLATION_PICKLE_PREFIX}-{col}-{ABLATION_PICKLE_SUFFIX}.pkl', 'rb') as f:
            print('ablating mlp ', col)
            x = pickle.load(f)
            x = x.cpu()
            df = pd.DataFrame(x, columns=['layer', 'neuron'])
            for idx, row in df.iterrows():
                model.model.layers[int(row['layer'])].mlp.down_proj.weight[int(row['neuron'])] = torch.zeros_like(model.model.layers[int(row['layer'])].mlp.down_proj.weight[int(row['neuron'])])
    
        with open(f'{PREFIX}/broca/{MODEL_NAME}/{MODEL_NAME}-attr-patch-scripts/attn/{ABLATION_PICKLE_PREFIX}-{col}-{ABLATION_PICKLE_SUFFIX}.pkl', 'rb') as f:
            print('ablating attn ', col)
            x = pickle.load(f)
            x = x.cpu()
            df = pd.DataFrame(x, columns=['layer', 'neuron'])
            for idx, row in df.iterrows():
                model.model.layers[int(row['layer'])].self_attn.o_proj.weight[int(row['neuron'])] = torch.zeros_like(model.model.layers[int(row['layer'])].self_attn.o_proj.weight[int(row['neuron'])])
    
    with torch.no_grad():
        ablate_model(col)

def parse_answer(text):
    answers = []
    for t in text:
        ans = t.split("A:")[-1].strip()
        answers.append(ans)
    return answers

def construct_prompt(train_dataset, num_demonstrations):
    assert num_demonstrations > 0
    prompt = ''
    train_examples = train_dataset.shuffle(seed=42).select(range(num_demonstrations))
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
def get_aligned_words_measures(texts: str, 
                               answers: str,
                               measure: str,
                               model: GPT2LMHeadModel, 
                               tokenizer: GPT2Tokenizer) -> list[str]:
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
datasets = {}
datasets[col] = Dataset.from_pandas(pd.DataFrame(df[[col, 'ng-' + col]].copy())).train_test_split(test_size=0.5, random_state=42)

def get_master_prompt(lang):
    en_verbs = ["affirms", "bring", "brings", "carries", "carry", "climb", "climbs", "eat", "eats", "hold", "holds", "knows", "notices", "read", "reads", "says", "sees", "take", "takes"]
    en_verbs_past = ["affirmed", "brought", "carried", "climbed", "ate", "held", "knew", "noticed", "read", "said", "saw", "took"]
    en_verbs_infinitive = ["to affirm", "to bring", "to carry", "to climb", "to eat", "to hold", "to know", "to notice", "to read", "to say", "to see", "to take"]
    en_verbs_passive = ["affirmed", "brought", "carried", "climbed", "eaten", "held", "known", "noticed", "read", "said", "seen", "taken"]
    en_nouns = ['author', 'banana', 'biscuit', 'book', 'bottle', 'box', 'boy', 'bulb', 'cap', 'cat', 'chalk', 'chapter', 'cucumber', 'cup', 'dog', 'fish', 'fruit', 'girl', 'hill', 'man', 'meal', 'mountain', 'mouse', 'newspaper', 'pear', 'pizza', 'poem', 'poet', 'rock', 'roof', 'speaker', 'staircase', 'story', 'teacher', 'toy', 'tree', 'woman', 'writer']
    en_nouns_plural = ['authors', 'boys', 'cats', 'dogs', 'girls', 'men', 'poets', 'speakers', 'teachers', 'women', 'writers']
    proper_nouns = ['Gomu', 'Harry', 'John', 'Leela', 'Maria', 'Sheela', 'Tom']
    
    ita_verbs = ['legge', 'leggono', 'mangia', 'mangiano', 'porta', 'portano', 'prende', 'prendono',
                'salgono', 'scala', 'tengono', 'tiene', 'vede', 'dice', 'osserva', 'sa', 'afferma']
    ita_verbs_past = ['è letto', 'è mangiato', 'è portato', 'è preso', 'è scalata', 'è scalato', 'è tenuto', 'è salito']
    ita_verbs_infinitive = ['leggere', 'magiare', 'portare', 'prendere', 'scalare', 'tenere', 'salire']
    ita_nouns = ['albero', 'autore', 'banana', 'biscotto', 'bottiglia', 'cane', 'capitolo', 'cappello', 'cetriolo', 'collina', 'donna', 'frutta', 'gatto', 'gesso', 'giocattolo', 'giornale', 'insegnante', 'lampadina', 'libro', 'montagna', 'oratorio', 'pasto', 'pera', 'pesce', 'pizza', 'poema', 'poeta', 'ragazza', 'ragazzo', 'roccia', 'scala', 'scatola', 'scrittore', 'storia', 'tazza', 'tetto', 'topo', 'uomo']
    ita_nouns_plurals = ['alberi', 'autori', 'banane', 'biscotti', 'bottiglie', 'cani', 'capitoli', 'cappelli', 'cetrioli', 'colline', 'donne', 'frutta', 'gatti', 'gessi', 'giocattoli', 'giornali', 'insegnanti', 'lampadine', 'libri', 'montagne', 'oratori', 'pasti', 'pere', 'pesci', 'pizze', 'poemi', 'poeti', 'ragazze', 'ragazzi', 'rocce', 'scale', 'scatole', 'scrittori', 'storie', 'tazze', 'tetti', 'topi', 'uomini']
    ita_nouns_la = ["pera", "banana", "bottiglia", "scatola", "lampadina", "credenza", "tazza", "frutta", "ragazza", "collina", "montagna", "pizza", "roccia", "scala", "storia", "donna"]
    ita_nouns_lo = ["scrittore"]
    ita_nouns_il = ["biscotto", "libro", "ragazzo", "cappello", "gatto", "capitolo", "gesso", "cetriolo", "cane", "oratorio", "pesce", "pasto", "topo", "giornale", "poeta", "poema", "tetto", "giocattolo"]
    ita_nouns_gli = ["scrittori", "uomini", "oratori", "insegnanti", "autori"]
    ita_nouns_i = ["ragazzi", "gatti", "cani", "poeti"]
    ita_nouns_il_vowel = ["uomo", "oratore", "insegnante", "albero", "autore"]
    ita_nouns_le = ["donne"]
    
    it_nouns_kon = ['author', 'biscuit', 'book', 'boy', 'cap', 'cat', 'chalk', 'chapter', 'cucumber', 'dog', 'fish', 'man', 'meal', 'mouse', 'newspaper', 'poet', 'rock', 'roof', 'speaker', 'teacher', 'toy', 'writer']
    it_nouns_kar = ['banana', 'bottle', 'box', 'bulb', 'cabinet', 'cup', 'fruit', 'girl', 'hill', 'mountain', 'pear', 'pizza', 'poem', 'staircase', 'story', 'tree', 'woman']
    it_nouns_kons = ['authors', 'boys', 'cats', 'dogs', 'men', 'poets', 'speakers', 'teachers', 'writers']
    it_nouns_kars = ["girls", "women"]
    jp_nouns = ["梨", "著者", "バナナ", "ビスケット", "本", "ボトル", "箱", "男の子", "電球", "帽子", "猫", "章", "白亜", "コップ", "胡瓜", "犬", "魚", "果物", "女の子", "丘", "男", "食事", "山", "マウス", "新聞", "麺", "詩人", "詩", "岩石", "屋根", "スピーカー", "階段", "小説", "先生", "玩具", "木", "女", "著者", "ピザ", "梨", "著者", "バナナ", "ビスケット", "本", "ボトル", "箱", "男の子", "電球", "帽子", "猫", "章", "白亜", "コップ", "胡瓜", "犬", "魚", "果物", "女の子", "丘", "男性", "食事", "山", "マウス", "新聞", "麺", "詩人", "詩", "岩石", "屋根", "スピーカー", "階段", "小説", "先生", "玩具", "木", "女性", "著者", "ピザ"]
    jp_verbs = ["食べる","読む","運ぶ","のぼる","とる","持つ","もたらす", "食べる","読む","運ぶ","のぼる","とる","持つ","もたらす"]
    jp_verbs_passive = ["食べられる", "読まれる", "運ばれる", "のぼられる", "とられる", "持たれる", "持たれる"]
    jp_suffixes = [ "は", "が", "を", "と", "に", "ない", "た" ]
    jp_proper_nouns = ["シーラ", "ゴム", "ハリー", "ジョン", "リーラ", "マリア", "トム"]

    if 'en' == "".join(lang[:2]):
        intro = "We will give you examples of English sentences that follow or violate the rules of a shared grammar, along with labels 'Yes' or 'No'. You will then generate a label, 'Yes' or 'No', for a new unlabeled sentence that may follow or violate the same grammar rules."
        # verbs = f"""The sentences may use verbs ({', '.join(en_verbs)});"""
        # pastTenseVerbs = f"""or their corresponding past tense forms ({', '.join(en_verbs_past)});""" 
        # infinitiveVerbs = f"""infinitive forms ({', '.join(en_verbs_infinitive)});"""
        # passiveVerbs = f"""or passive forms ({', '.join(en_verbs_passive)})."""
        # nouns = f"""The sentences may use nouns ({', '.join(set(en_nouns + en_nouns_plural))}) for the subjects and objects."""
        # properNouns = f"""The sentences may use proper nouns ({', '.join(set(proper_nouns))})."""
        # return f"""1.{intro}\n2.{verbs} {pastTenseVerbs} {infinitiveVerbs} {passiveVerbs}\n3.{nouns}\n4.{properNouns}"""
        return f"{intro}"

    elif 'it' == "".join(lang[:2]) and (len(lang) <=2 or lang[2] != 'a'):
        intro = "We will give you examples of English sentences stylized to Italian syntax that follow or violate the rules of a shared grammar, along with labels 'Yes' or 'No'. You will then generate a label, 'Yes' or 'No', for a new unlabeled sentence that may follow or violate the same grammar rules."
        # verbs = f"""The sentences may use verbs ({', '.join(en_verbs)});"""
        # pastTenseVerbs = f"""or their corresponding past tense forms ({', '.join(en_verbs_past)});""" 
        # infinitiveVerbs = f"""infinitive forms ({', '.join(en_verbs_infinitive)});"""
        # passiveVerbs = f"""or passive forms ({', '.join(en_verbs_passive)})."""
        # nouns = f"""The sentences may use nouns ({', '.join(set(en_nouns + en_nouns_plural))}) for the subjects and objects."""
        # properNouns = f"""The sentences may use proper nouns ({', '.join(set(proper_nouns))})."""
        gendered = f"""The nouns in the sentences have specific gender determiners - 'kar' (used by {', '.join(set(it_nouns_kar))}); 'kon' (used by {', '.join(set(it_nouns_kon))}); 'kars' (used by {', '.join(set(it_nouns_kars))}); 'kons' (used by {', '.join(set(it_nouns_kons))})."""
        # return f"""1.{intro}\n2.{verbs} {pastTenseVerbs} {infinitiveVerbs} {passiveVerbs}\n3.{nouns}\n4.{properNouns}\n5.{gendered}"""
        return f"1. {intro}\n2. {gendered}"
    
    elif 'ita' == "".join(lang[:3]):
        intro = "We will give you examples of Italian sentences that follow or violate the rules of a shared grammar, along with labels 'Yes' or 'No'. You will then generate a label, 'Yes' or 'No', for a new unlabeled sentence that may follow or violate the same grammar rules."
        # verbs = f"""The sentences may use verbs ({', '.join(ita_verbs)});"""
        # pastTenseVerbs = f"""or their corresponding past tense forms ({', '.join(ita_verbs_past)});""" 
        # infinitiveVerbs = f"""infinitive forms ({', '.join(ita_verbs_infinitive)});"""
        # passiveVerbs = f"""or passive forms ({', '.join(ita_verbs_past)})."""
        # nouns = f"""The sentences may use nouns ({', '.join(set(ita_nouns + ita_nouns_plurals))}) for the subjects and objects."""
        # properNouns = f"""The sentences may use proper nouns ({', '.join(set(proper_nouns))})."""
        # gendered = f"""The nouns in the sentences have specific gender determiners - 'la' (used by {', '.join(set(ita_nouns_la))}); 'lo' (used by {', '.join(set(ita_nouns_lo))}); 'le' (used by {', '.join(set(ita_nouns_le))}); 'il' (used by {', '.join(set(ita_nouns_il))}, "l'" (used by {', '.join(set(ita_nouns_il_vowel))}); 'i' (used by {', '.join(set(ita_nouns_i))}, and 'gli' (used by {', '.join(set(ita_nouns_gli))}."""
        # return f"""1.{intro}\n2.{verbs} {pastTenseVerbs} {infinitiveVerbs} {passiveVerbs}\n3.{nouns}\n4.{properNouns}\n5.{gendered}"""
        return f"{intro}"
    
    elif 'jap' == "".join(lang[:3]):
        intro = "We will give you examples of Japanese sentences that follow or violate the rules of a shared grammar, along with labels 'Yes' or 'No'. You will then generate a label, 'Yes' or 'No', for a new unlabeled sentence that may follow or violate the same grammar rules."
        # verbs = f"""The sentences may use verbs ({', '.join(jp_verbs)});"""
        # passiveVerbs = f"""or passive forms ({', '.join(jp_verbs_passive)})."""
        # nouns = f"""The sentences may use nouns ({', '.join(set(jp_nouns))}) for the subjects and objects."""
        # suffixes = f"""The sentences may use suffixes ({', '.join(set(jp_suffixes))}) along with subjects, objects or verbs."""
        # properNouns = f"""The sentences may use proper nouns ({', '.join(set(jp_proper_nouns))})."""
        # return f"""1.{intro}\n2.{verbs} {passiveVerbs}\n3.{nouns}\n4.{properNouns}\n5.{suffixes}"""
        return f"{intro}"
    
    elif 'jp' == "".join(lang[:2]):
        intro = "We will give you examples of English sentences stylized to Japanese syntax that follow or violate the rules of a shared grammar, along with labels 'Yes' or 'No'. You will then generate a label, 'Yes' or 'No', for a new unlabeled sentence that may follow or violate the same grammar rules."
        # verbs = f"""The sentences may use verbs ({', '.join(en_verbs)});"""
        # pastTenseVerbs = f"""or their corresponding past tense forms ({', '.join(en_verbs_past)});""" 
        # infinitiveVerbs = f"""infinitive forms ({', '.join(en_verbs_infinitive)});"""
        # passiveVerbs = f"""or passive forms ({', '.join(en_verbs_passive)});"""
        # nouns = f"""The sentences may use nouns ({', '.join(set(en_nouns + en_nouns_plural))}) for the subjects and objects."""
        # properNouns = f"""The sentences may use proper nouns ({', '.join(set(proper_nouns))})."""
        suffixes = f"""The sentences use Japanese topic markers and suffixes such as wa (commonly used after the subject); o, ni, ga, o-ta (commonly used after the object); reru(used after the verb)."""
        # return f"""1.{intro}\n2.{verbs} {pastTenseVerbs} {infinitiveVerbs} {passiveVerbs}\n3.{nouns}\n4.{properNouns}\n5.{suffixes}"""
        return f"1. {intro}\n2. {suffixes}"

master_prompt = get_master_prompt(col)
train_dataset = datasets[col]['train']
test_dataset = datasets[col]['test']
printAnswer = False
f = pd.DataFrame(columns=["type", "prompt", "q", "prediction", "gold", "surprisal", "int-grad"])
for i in range(0, len(test_dataset), BATCH_SIZE):
    if (i > 10):
        break
    test_sentences = []
    fPrompts = []
    fQs = []
    fGolds = []
    prompts = []
    for batch_idx in range(BATCH_SIZE):
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
        f = pd.concat([f, pd.DataFrame([{'type': col, 'prompt': prompts[batch_idx], 'q' :test_sentences[batch_idx], 'prediction': fPredictions[batch_idx], 'gold': fGolds[batch_idx], 'surprisal': fSurprisals[batch_idx], 'int-grad': 0}])]).reset_index(drop=True)
# Evaluate
accuracy = compute_accuracy(preds, golds)
print(f"{col} -- Accuracy: {accuracy:.2f}\n")
g = pd.concat([g, pd.DataFrame([{ 'trainType' : col, 'testType': col, 'accuracy': f"{accuracy:.2f}"}])])
f.to_csv(f"{PREFIX}/broca/{MODEL_NAME}/experiments/{col}.csv")
g.to_csv(f'{PREFIX}/broca/{MODEL_NAME}/experiments/{col}.csv')
