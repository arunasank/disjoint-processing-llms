from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig,
    BitsAndBytesConfig
)
import bitsandbytes
from accelerate import infer_auto_device_map
from nnsight import LanguageModel
import sys
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import Dataset, load_dataset
import torch
import random
import pandas as pd
from tqdm import tqdm
import os
import pickle
import yaml
import argparse
import numpy as np
random.seed(42)
torch.manual_seed(42)

parser = argparse.ArgumentParser()

parser.add_argument('--config', type=str, help='path to the model training config file, found in broca/configs')
parser.add_argument('--stype', type=int, help='grammar structure col number, found in broca/data-gen')

args = parser.parse_args()
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

# args = { "config": "/mnt/align4_drive/arunas/broca/configs/mistral-icl-config", "stype": 7 }
# with open(args["config"], 'r') as f:
#     config = yaml.safe_load(f)

PREFIX = config["prefix"]
MODEL_NAME = config["model_name"]
MODEL_PATH = config["model_path"]
ABLATION = config["ablation"]
DATA_PATH = config["data_path"]
NUM_DEMONSTRATIONS = config["num_dems"]
BATCH_SIZE = config["batch_size"]
FINAL_CSV_SUBPATH = config["final_csv_subpath"]
MAX_LEN = 0

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

df = pd.read_csv(f'{DATA_PATH}')
gCols = [col for col in list(df.columns) if not 'ng' in col]

# col = gCols[args["stype"]]

col = gCols[args.stype]

df = pd.read_csv(f'{DATA_PATH}')

if (ABLATION):
    print('ABLATION!')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, device_map="auto", padding_side="left")
    
    tokenizer.pad_token = tokenizer.eos_token
    model = LanguageModel(MODEL_PATH, device_map='auto') # Load the model
    
    device_map = infer_auto_device_map(model)
    MEAN_PICKLES_PATH = config["mean_pickles_path"]
    MEAN_PICKLES_SUBPATH = config["mean_pickles_subpath"]
    PATCH_PICKLES_PATH = config["patch_pickles_path"]
    PATCH_PICKLES_SUBPATH = config["patch_pickles_subpath"]
    TOPK = config['topk']
    RANDOMLY_SAMPLE = config['randomly_sample']
    ABLATE_UNION = config['ablate_union']
    ABLATE_INTERSECTION = config['ablate_intersection']
    NUM_HIDDEN_STATES = config['num_hidden_states']
    
    def retrieve_topK(col, component, topK):
        with open(f'{PATCH_PICKLES_PATH}/{component}/{PATCH_PICKLES_SUBPATH}/{col}.pkl', 'rb') as f:
            print(f'ablating {component} ', col)
            component_cache = pickle.load(f)
            print('before top k', component_cache.shape)
            component_cache = component_cache.cpu()
            flattened_effects_cache = component_cache.view(-1)
            top_neurons = flattened_effects_cache.topk(k=int(topK * flattened_effects_cache.shape[-1]))
            two_d_indices = torch.cat((((top_neurons[1] // component_cache.shape[1]).unsqueeze(1)), ((top_neurons[1] % component_cache.shape[1]).unsqueeze(1))), dim=1)            
            df = pd.DataFrame(two_d_indices, columns=['layer', 'neuron'])
        return df
    
    def retrieve_randomK(col, component, TOPK):
        df = retrieve_topK(col, component, TOPK)
        layer_counts = df['layer'].value_counts()
        random_df = pd.DataFrame(columns=['layer', 'neuron'])
        for layer, count in layer_counts.items():
            max_neuron_index = NUM_HIDDEN_STATES
            sampled_neurons = np.random.choice(max_neuron_index, size=count, replace=False)
            temp_df = pd.DataFrame({'layer': [layer] * count, 'neuron': sampled_neurons})
            random_df = pd.concat([random_df, temp_df], ignore_index=True)
        return random_df
    
    def retrieve_union(component, TOPK):
        if os.path.exists(f'{PATCH_PICKLES_PATH}/{component}/{PATCH_PICKLES_SUBPATH}/union.csv'):
            union_df = pd.read_csv(f'{PATCH_PICKLES_PATH}/{component}/{PATCH_PICKLES_SUBPATH}/union.csv')    
        else:
            union_df = pd.DataFrame(columns=['layer', 'neuron'])
            for col in gCols:
                try:
                    df = retrieve_topK(col, component, TOPK)
                    union_df = pd.concat([union_df, df], axis=1)
                except:
                    print(col)
            union_df.drop_duplicates(inplace=True)
            union_df.to_csv(f'{PATCH_PICKLES_PATH}/{component}/{PATCH_PICKLES_SUBPATH}/union.csv')
        return union_df

    def retrieve_intersection(component, TOPK):
        
        def find_intersection(df): # double check, from chaatgpt. I ran a few tests, but needs careful understanding.
            # Split the DataFrame into groups based on the 'type'
            grouped = df.groupby('type')

            # Initialize a set with the first type's (layer, neuron) tuples
            first_type = next(iter(grouped.groups.keys()))
            intersection_set = set(tuple(row) for row in grouped.get_group(first_type)[['layer', 'neuron']].values)

            # Perform intersection with other types
            for name, group in grouped:
                current_set = set(tuple(row) for row in group[['layer', 'neuron']].values)
                intersection_set.intersection_update(current_set)

            # If intersection set is not empty, convert it back to a DataFrame
            if intersection_set:
                result_df = pd.DataFrame(list(intersection_set), columns=['layer', 'neuron'])
            else:
                result_df = pd.DataFrame(columns=['layer', 'neuron'])

            return result_df

        if os.path.exists(f'{PATCH_PICKLES_PATH}/{component}/{PATCH_PICKLES_SUBPATH}/intersection.csv'):
            intersection_df = pd.read_csv(f'{PATCH_PICKLES_PATH}/{component}/{PATCH_PICKLES_SUBPATH}/intersection.csv')    
        else:
            all_df = pd.DataFrame(columns=['type', 'layer', 'neuron'])
            for col in gCols:
                try:
                    df = retrieve_topK(col, component, TOPK)
                    df['type'] = col
                    all_df = pd.concat([all_df, df], axis=1)
                except:
                    print(col)
            intersection_df = find_intersection(all_df)
            intersection_df.to_csv(f'{PATCH_PICKLES_PATH}/{component}/{PATCH_PICKLES_SUBPATH}/intersection.csv')
        return intersection_df

    def ablation_cache(col, component):
        global MAX_LEN
        if (RANDOMLY_SAMPLE):
            print('### RANDOMLY SAMPLE NEURONS FOR ABLATION ')
            df = retrieve_randomK(col, component, TOPK)
        elif (ABLATE_UNION):
            print('### ABLATE UNION OF TOPK NEURONS FROM ALL LANGUAGES ')
            df = retrieve_union(component, TOPK)
        elif (ABLATE_INTERSECTION):
            print('### ABLATE INTERSECTION OF TOPK NEURONS FROM ALL LANGUAGES ')
            df = retrieve_union(component, TOPK)
        else:
            df = retrieve_topK(col, component, TOPK)
        with open(f'{MEAN_PICKLES_PATH}/{component}/{MEAN_PICKLES_SUBPATH}/{col}.pkl', 'rb') as mf:
            component_cache = pickle.load(mf)
            component_cache = component_cache.cpu()
            comp_values = []
            for idx, row in df.iterrows():
                comp_values.append(list(component_cache[row['layer'], :, row['neuron']].numpy().flatten()))
            MAX_LEN = len(comp_values[0])
            df['values'] = comp_values
        return df

    with torch.no_grad():
        mlp_ablate = ablation_cache(col, 'mlp')
        attn_ablate = ablation_cache(col, 'attn')
else:
    model_config = AutoConfig.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, config=model_config, device_map="auto", padding_side="left")
    
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, config=model_config, device_map='auto') # Load the model
    
    device_map = infer_auto_device_map(model)
    
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
    print(len(preds), len(golds))
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
        datas.append(data)
    return datas

preds = []
golds = []

f = pd.DataFrame(columns=["type", "prompt", "q", "prediction", "gold", "surprisal", "int-grad"])
f['type'] = 'test'
g = pd.DataFrame(columns=['accuracy', 'type'])
datasets = {}
np.random.seed(42)
datasets[col] = Dataset.from_pandas(pd.DataFrame(df[[col, 'ng-' + col]].copy())).train_test_split(test_size=0.5)
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
    
if (not (os.path.exists(f"{PREFIX}/broca/{MODEL_NAME}/experiments/{FINAL_CSV_SUBPATH}/{col}.csv")) and not (os.path.exists(f"{PREFIX}/broca/{MODEL_NAME}/experiments/{FINAL_CSV_SUBPATH}/{col}-acc.csv"))):
    master_prompt = get_master_prompt(col)
    train_dataset = datasets[col]['train']
    test_dataset = datasets[col]['test']
    printAnswer = False
    f = pd.DataFrame(columns=["type", "prompt", "q", "prediction", "gold", "surprisal", "int-grad"])
    for i in tqdm(range(0, len(test_dataset), BATCH_SIZE)):
        test_sentences = []
        fPrompts = []
        fQs = []
        fGolds = []
        prompts = []
        for batch_idx in range(min(BATCH_SIZE, len(test_dataset) - i)):
            testBadOrGood = random.choice(['ng-', ''])
            test_sentence = test_dataset[i + batch_idx]
            prompt = construct_prompt(train_dataset, NUM_DEMONSTRATIONS)
            
            fPrompt = prompt
            
            # Append test example
            prompt += "Q: Is this sentence grammatical? Yes or No: "
            prompt += test_sentence[testBadOrGood + col]
            prompt += "\nA: "
            
            fQ = "Q: Is this sentence grammatical? Yes or No: " + test_sentence[testBadOrGood + col] + "\nA:"
            
            if testBadOrGood == 'ng-':
                golds.append("No")
                fGold = 'No'
            else:
                golds.append("Yes")
                fGold = 'Yes'
            fGolds.append(fGold)
            prompts.append(f'{master_prompt}\n\n{prompt}')
            test_sentences.append(test_sentence[testBadOrGood + col])
            fPrompts.append(fPrompt)
            fQs.append(fQ)
        answers = []
        if (ABLATION):
            for prompt in prompts:
                prompt = tokenizer.decode(tokenizer(prompt, padding='max_length', max_length=MAX_LEN)["input_ids"])
                with model.trace(prompt, scan=False, validate=False) as tracer:
                    for idx, row in mlp_ablate.iterrows():
                        model.model.layers[row['layer']].mlp.down_proj.output[0, :len(row['values']), row['neuron']] = torch.tensor(row['values'])
                    for idx, row in attn_ablate.iterrows():
                        model.model.layers[row['layer']].self_attn.o_proj.output[0, :len(row['values']), row['neuron']] = torch.tensor(row['values'])
                    token_ids = model.lm_head.output.argmax(dim=-1).save()
                answers.append(model.tokenizer.decode(token_ids[0][-1]))
                    
            preds = preds + parse_answer(answers)
            fPredictions = parse_answer(answers)
            for batch_idx in range(len(fPrompts)):
                f = pd.concat([f, pd.DataFrame([{'type': col, 'prompt': prompts[batch_idx], 'q' :test_sentences[batch_idx], 'prediction': fPredictions[batch_idx], 'gold': fGolds[batch_idx], 'int-grad': 0}])]).reset_index(drop=True)
        else:
        # Get answer from model
            model_inputs = tokenizer(prompts, return_tensors="pt", padding=True).to('cuda')
            answers = model.generate(**model_inputs, pad_token_id=tokenizer.eos_token_id, max_new_tokens=2, top_p=0.9, temperature=0.1, do_sample=True)
            answers = tokenizer.batch_decode(answers)[:BATCH_SIZE]
            preds = preds + parse_answer(answers)
            fPredictions = parse_answer(answers)
            fSurprisals = get_aligned_words_measures(test_sentences, parse_answer(answers), "surp", model, tokenizer)
            for batch_idx in range(len(fPrompts)):
                f = pd.concat([f, pd.DataFrame([{'type': col, 'prompt': prompts[batch_idx], 'q' :test_sentences[batch_idx], 'prediction': fPredictions[batch_idx], 'gold': fGolds[batch_idx], 'surprisal': fSurprisals[batch_idx], 'int-grad': 0}])]).reset_index(drop=True)
    # Evaluate
    accuracy = compute_accuracy(preds, golds)
    print(f"{col} -- Accuracy: {accuracy:.2f}\n")
    g = pd.concat([g, pd.DataFrame([{ 'trainType' : col, 'testType': col, 'accuracy': f"{accuracy:.2f}"}])])
    f.to_csv(f"{PREFIX}/broca/{MODEL_NAME}/experiments/{FINAL_CSV_SUBPATH}/{col}.csv")
    g.to_csv(f'{PREFIX}/broca/{MODEL_NAME}/experiments/{FINAL_CSV_SUBPATH}/{col}-acc.csv', index=False)
