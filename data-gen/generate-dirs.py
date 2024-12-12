import pandas as pd
import os 
import torch
import random
import numpy as np
from datasets import Dataset
from tqdm import tqdm
import json
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
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

df = pd.read_csv(f'/mnt/align4_drive/arunas/broca/data-gen/ngs.csv')
gCols = [col for col in sorted(df.columns) if not 'ng' in col[:2]]

for col in tqdm(gCols):
    path = os.path.join('/mnt/align4_drive/arunas/broca/data-gen/new/', col)
    os.makedirs(path, exist_ok=True) 
    with open(f"{path}/prompts.txt", "w+") as promptFile:
        promptFile.write(get_master_prompt(col))
        
    datasets = Dataset.from_pandas(pd.DataFrame(df[[col, 'ng-' + col]].copy())).train_test_split(test_size=0.5)
    train_dataset = datasets['train']
    test_dataset = datasets['test']
    
    with open(f"{path}/train.json", "a+") as train:
        for row in train_dataset:
            train.write(json.dumps({"sentence": row[col], "label": "Yes"}) + "\n")
            train.write(json.dumps({"sentence": row[f"ng-{col}"], "label": "No"}) + "\n")
            
    with open(f"{path}/test.json", "a+") as test:
        for row in test_dataset:
            test.write(json.dumps({"sentence": row[col], "label": "Yes"}) + "\n")
            test.write(json.dumps({"sentence": row[f"ng-{col}"], "label": "No"}) + "\n")
    
    
        
        



