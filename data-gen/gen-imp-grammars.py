import pandas as pd
import re
import random
random.seed(42)
from tqdm.auto import tqdm
tqdm.pandas()


## Genders for subjects based on italian


genderDict = {
    "pear": "kar",
    "author": "kon",
    "authors": "kons",
    "banana": "kar",
    "biscuit": "kon",
    "book": "kon",
    "bottle": "kar",
    "box": "kar",
    "boy": "kon",
    "boys": "kons",
    "bulb": "kar",
    "cabinet": "kar",
    "cap": "kon",
    "cat": "kon",
    "cats": "kons",
    "chapter": "kon",
    "chalk": "kon",
    "cup": "kar",
    "cucumber": "kon",
    "dog": "kon",
    "dogs": "kons",
    "fish": "kon",
    "fruit": "kar",
    "girl": "kar",
    "girls": "kars",
    "hill": "kar",
    "man": "kon",
    "men": "kons",
    "meal": "kon",
    "mountain": "kar",
    "mouse": "kon",
    "newspaper": "kon",
    "pizza": "kar",
    "poet": "kon",
    "poets": "kons",
    "poem": "kar",
    "rock": "kon",
    "roof": "kon",
    "speaker": "kon",
    "speakers": "kons",
    "staircase": "kar",
    "story": "kar",
    "teacher": "kon",
    "teachers": "kons",
    "toy": "kon",
    "tree": "kar",
    "woman": "kar",
    "women": "kars",
    "writer": "kon",
    "writers": "kons"
}

pastTense = {
    'climbs' : 'climbed',
    'reads': 'read',
    'carries': 'carried',
    'eats': 'ate',
    'holds': 'held',
    'takes' :'took',
    'brings': 'brought',
    'reads': 'read',
    'climb' : 'climbed',
    'read': 'read',
    'carry': 'carried',
    'eat': 'ate',
    'hold': 'held',
    'take' :'took',
    'bring': 'brought',
    'read': 'read'
}

infinitive = {
    'climbs' : 'to climb',
    'reads': 'to read',
    'carries': 'to carry',
    'eats': 'to eat',
    'holds': 'to hold',
    'takes' : 'to take',
    'brings': 'to bring',
    'reads': 'to read',
    'climb' : 'to climb',
    'read': 'to read',
    'carry': 'to carry',
    'eat': 'to eat',
    'hold': 'to hold',
    'take' : 'to take',
    'bring': 'to bring',
    'read': 'to read'
}

pluralObjects = {
    'fish': 'fish',
    'mouse': 'mice',
    'bottle': 'bottles',
    'newspaper': 'newspapers',
    'chalk': 'chalks',
    'box': 'boxes',
    'cap': 'caps',
    'bulb': 'bulbs',
    'cup': 'cups',
    'toy': 'toys',
    'staircase': 'staircases',
    'rock': 'rocks',
    'hill': 'hills',
    'mountain': 'mountains',
    'roof': 'roofs',
    'tree': 'trees',
    'biscuit': 'biscuits',
    'banana': 'bananas',
    'pear': 'pears',
    'meal': 'meals',
    'fruit': 'fruits',
    'cucumber': 'cucumbers',
    'pizza': 'pizzas',
    'book': 'books',
    'poem': 'poems',
    'story': 'stories',
    'chapter': 'chapters'
}

passiveSeed = {
    'carries': 'carried',
    'carry': 'carried',
    'holds': 'held',
    'hold': 'held',
    'takes': 'taken',
    'take': 'taken',
    'brings': 'brought',
    'bring': 'brought',
    'climbs': 'climbed',
    'climb': 'climbed',
    'eats': 'eaten',
    'eat': 'eaten',
    'reads': 'read',
    'read': 'read'
}

it_genderDict = {
    "pera": ["la", "una"],
    "scrittore": ["lo", "uno"],
    "scrittori": ["gli"],
    "banana": ["la", "una"],
    "biscotto": ["il", "un"],
    "libro": ["il", "un"],
    "bottiglia": ["la", "una"],
    "scatola": ["la", "una"],
    "ragazzo": ["il", "un"],
    "ragazzi": ["i"],
    "lampadina": ["la", "una"],
    "credenza": ["la", "una"],
    "cappello": ["il", "un"],
    "gatto": ["il", "un"],
    "gatti": ["i"],
    "capitolo": ["il", "un"],
    "gesso": ["il", "un"],
    "tazza": ["la", "una"],
    "cetriolo": ["il", "un"],
    "cane": ["il", "un"],
    "cani": ["i"],
    "oratorio": ["il", "un"],
    "pesce": ["il", "un"],
    "frutta": ["la", "una"],
    "ragazza": ["la", "una"],
    "ragazze": ["le"],
    "collina": ["la", "una"],
    "uomo": ["l'", "un'"],
    "uomini": ["gli"],
    "pasto": ["il", "un"],
    "montagna": ["la", "una"],
    "topo": ["il", "un"],
    "giornale": ["il", "un"],
    "pizza": ["la", "una"],
    "poeta": ["il", "un"],
    "poeti": ["i"],
    "poema": ["il", "un"],
    "roccia": ["la", "una"],
    "tetto": ["il", "un"],
    "oratore": ["l'", "un'"],
    "oratori": ["gli"],
    "scala": ["la", "una"],
    "storia": ["la", "una"],
    "insegnante": ["l'", "un'"],
    "insegnanti": ["gli"],
    "giocattolo": ["il", "un"],
    "albero": ["l'", "un'"],
    "donna": ["la", "una"],
    "donne": ["le"],
    "autore": ["l'", "un'"],
    "autori": ["gli"]
}

it_pastTense = {
    'scala' : { 'la': 'ha scalata', 'il': 'ha scalato'},
    'legge': { 'la': 'ha letto', 'il': 'ha letto' },
    'porta': { 'la': 'ha portato', 'il': 'ha portato' },
    'mangia': { 'la': 'ha mangiato', 'il': 'ha mangiato' },
    'tiene': { 'la': 'ha tenuto', 'il': 'ha tenuto' },
    'prende' : { 'la': 'ha preso', 'il': 'ha preso'}
}

it_infinitive = {
    'scala' : 'scalare',
    'legge': 'leggere',
    'porta': 'portare',
    'mangia': 'mangiare',
    'tiene': 'tenere',
    'prende' : 'prendere'
}

it_pluralObjects = {
    'pesce': 'pesci',
    'topo': 'topi',
    'bottiglia': 'bottiglie',
    'giornale': 'giornali',
    'gesso': 'gessi',
    'scatola': 'scatole',
    'cappello': 'cappelli',
    'lampadina': 'lampadine',
    'tazza': 'tazze',
    'giocattolo': 'giocattoli',
    'scala': 'scale',
    'roccia': 'rocce',
    'collina': 'colline',
    'montagna': 'montagne',
    'tetto': 'tetti',
    'albero': 'alberi',
    'biscotto': 'biscotti',
    'banana': 'banane',
    'pera': 'pere',
    'pasto': 'pasti',
    'frutta': 'frutta',
    'cetriolo': 'cetrioli',
    'pizza': 'pizze',
    'libro': 'libri',
    'poema': 'poemi',
    'storia': 'storie',
    'capitolo': 'capitolo'
}

it_passiveSeed = {
    'scala' : { 'la': 'è scalata', 'una': 'è scalata', 'il': 'è scalato', 'un': 'è scalato', 'uno': 'è scalato'},
    'legge': { 'la': 'è letto', 'una': 'è letto', 'il': 'è letto' , 'un': 'è letto', 'uno': 'è letto' },
    'porta': { 'la': 'è portato', 'una': 'è portato', 'il': 'è portato' , 'un': 'è portato', 'uno': 'è portato' },
    'mangia': { 'la': 'è mangiato', 'una': 'è mangiato', 'il': 'è mangiato' , 'un': 'è mangiato', 'uno': 'è mangiato' },
    'tiene': { 'la': 'è tenuto', 'una': 'è tenuto', 'il': 'è tenuto' , 'un': 'è tenuto', 'uno': 'è tenuto' },
    'prende' : { 'la': 'è preso', 'una': 'è preso', 'il': 'è preso', 'un': 'è preso', 'uno': 'è preso'},
    'salgono' : { 'la': 'è scalata', 'una': 'è scalata', 'il': 'è scalato', 'un': 'è scalato', 'uno': 'è scalato'},
    'leggono': { 'la': 'è letto', 'una': 'è letto', 'il': 'è letto' , 'un': 'è letto', 'uno': 'è letto' },
    'portano': { 'la': 'è portato', 'una': 'è portato', 'il': 'è portato' , 'un': 'è portato', 'uno': 'è portato' },
    'mangiano': { 'la': 'è mangiato', 'una': 'è mangiato', 'il': 'è mangiato' , 'un': 'è mangiato', 'uno': 'è mangiato' },
    'tengono': { 'la': 'è tenuto', 'una': 'è tenuto', 'il': 'è tenuto' , 'un': 'è tenuto', 'uno': 'è tenuto' },
    'prendono' : { 'la': 'è preso', 'una': 'è preso', 'il': 'è preso', 'un': 'è preso', 'uno': 'è preso'}
}


## Generate sentences

seed = [ { 'verb' : ['carries', 'holds', 'takes', 'brings'],  'subject': ['dog', 'cat', 'man', 'woman', 'teacher', 'girl', 'boy'], 'object': ['fish', 'mouse', 'bottle', 'newspaper', 'chalk', 'box', 'cap', 'bulb', 'cup', 'toy']},

{ 'verb': ['climbs'], 'subject': ['dog', 'cat', 'man', 'woman', 'teacher', 'girl', 'boy'], 'object': ['staircase', 'rock', 'hill', 'mountain', 'roof', 'tree'] },

{ 'verb': ['eats'], 'subject' : ['dog', 'cat', 'man', 'woman', 'teacher', 'girl', 'boy'], 'object': ['biscuit', 'fish', 'banana', 'pear', 'meal', 'fruit', 'cucumber', 'pizza' ]},
{'verb': ['reads'], 'subject' : ['poet', 'author', 'writer', 'speaker', 'teacher', 'girl', 'boy'], 'object': ['book', 'poem', 'story', 'chapter']} ]

subordinateSeed = [ { 'verb' : ['sees', 'says', 'notices', 'states', 'claims'],  'subject': ['Sheela', 'Leela', 'Maria', 'Gomu', 'John', 'Tom', 'Harry'], }]


it_seed = [ { 'verb' : ['porta', 'tiene', 'prende', 'porta'],  'subject': ['cane', 'gatto', 'uomo', 'donna', 'insegnante', 'ragazza', 'ragazzo'], 'object': ['pesce', 'topo', 'bottiglia', 'giornale', 'gesso', 'scatola', 'cappello', 'lampadina', 'tazza', 'giocattolo']},

{ 'verb': ['scala'], 'subject': ['cane', 'gatto', 'uomo', 'donna', 'insegnante', 'ragazza', 'ragazzo'], 'object': ['scala', 'roccia', 'collina', 'montagna', 'tetto', 'albero'] },

{ 'verb': ['mangia'], 'subject' : ['cane', 'gatto', 'uomo', 'donna', 'insegnante', 'ragazza', 'ragazzo'], 'object': ['biscotto', 'pesce', 'banana', 'pera', 'pasto', 'frutta', 'cetriolo', 'pizza' ]},
{'verb': ['legge'], 'subject' : ['poeta', 'autore', 'scrittore', 'oratorio', 'insegnante', 'ragazza', 'ragazzo'], 'object': ['libro', 'poema', 'storia', 'capitolo']} ]

it_subordinateSeed = [ { 'verb' : ['vede', 'dice', 'osserva', 'sa', 'afferma'],  'subject': ['Sheela', 'Leela', 'Maria', 'Gomu', 'John', 'Tom', 'Harry'], }]

df = pd.DataFrame()

for oidx, obj in enumerate(seed):
    for sidx, subj in enumerate(obj['subject']):
        for obidx, ob in enumerate(obj['object']):
            for vidx, verb in enumerate(obj['verb']):
                sdet = random.choice(['the', 'a'])
                odet = random.choice(['the', 'a'])
                pSubj = random.choice(subordinateSeed[0]['subject'])
                pVerb = random.choice(subordinateSeed[0]['verb'])

                if sdet == 'the':
                    it_sdet = it_genderDict[it_seed[oidx]['subject'][sidx]][0]
                else:
                    it_sdet = it_genderDict[it_seed[oidx]['subject'][sidx]][1]

                if odet == 'the':
                    it_odet = it_genderDict[it_seed[oidx]['object'][obidx]][0]
                else:
                    it_odet = it_genderDict[it_seed[oidx]['object'][obidx]][1]

                it_subj = it_seed[oidx]['subject'][sidx]
                it_ob = it_seed[oidx]['object'][obidx]
                it_verb = it_seed[oidx]['verb'][vidx]

                it_pSubj = random.choice(it_subordinateSeed[0]['subject'])
                it_pVerb = random.choice(it_subordinateSeed[0]['verb'])

                temp_odet = it_odet
                if it_odet =='il':
                    ita_passive_from = 'dal'
                if it_odet =='la':
                    ita_passive_from = 'dalla'
                elif it_odet == "un" or it_odet == "un'":
                    ita_passive_from = 'da un'
                    temp_odet = 'un'
                elif it_odet == "una":
                    ita_passive_from = 'da una'
                elif it_odet == "l'":
                    temp_odet = 'il'
                    ita_passive_from = "dall'"
                ita_passive_sentence = f"{it_odet} {it_ob} {it_passiveSeed[it_verb][temp_odet]} {ita_passive_from} {it_subj}"

                if (it_sdet != "l'" and it_odet != "l'"):
                    df = pd.concat([df, pd.DataFrame.from_dict([{
                        "ita": f"{it_sdet} {it_subj} {it_verb} {it_odet} {it_ob}",
                        "ita-r-1-null_subject": f"{it_verb} {it_odet} {it_ob}",
                        "ita-r-2-subordinate": f"{it_pSubj} {it_pVerb} che {it_sdet} {it_subj} {it_verb} {it_odet} {it_ob}",
                        "ita-r-3-passive":ita_passive_sentence,
                        "ita-u-1-negation": f"{it_sdet} {it_subj} {it_verb} {it_odet} no {it_ob}",
                        "ita-u-2-invert": " ".join(f"{it_sdet} {it_subj} {it_verb} {it_odet} {it_ob}".split(" ")[::-1]),
                        "ita-u-3-gender":f"{it_odet} {it_subj} {it_verb} {it_odet} {it_ob}",
                        "en": f"{sdet} {subj} {verb} {odet} {ob}",
                        "en-r-1-subordinate": f"{pSubj} {pVerb} that the {subj} {verb} the {ob}",
                        "en-r-2-passive": f"{odet} {ob} is {passiveSeed[verb]} by {sdet} {subj}",
                        "en-u-1-negation": f"{sdet} {subj} {verb} {odet} doesn't {ob}",
                        "en-u-2-inversion": " ".join(f"{sdet} {subj} {verb} {odet} {ob}".split(" ")[::-1]),
                        "en-u-3-qsubordinate": f"{pSubj} {pVerb} that does the {subj} {verb} the {ob}?"
                    }])])
                elif (it_sdet == "l'" and it_odet != "l'"):
                    df = pd.concat([df, pd.DataFrame.from_dict([{
                        "ita": f"{it_sdet}{it_subj} {it_verb} {it_odet} {it_ob}",
                        "ita-r-1-null_subject": f"{it_verb} {it_odet} {it_ob}",
                        "ita-r-2-subordinate": f"{it_pSubj} {it_pVerb} che {it_sdet}{it_subj} {it_verb} {it_odet} {it_ob}",
                        "ita-r-3-passive":ita_passive_sentence,
                        "ita-u-1-negation": f"{it_sdet}{it_subj} {it_verb} {it_odet} no {it_ob}",
                        "ita-u-2-invert": " ".join(f"{it_sdet}{it_subj} {it_verb} {it_odet} {it_ob}".split(" ")[::-1]),
                        "ita-u-3-gender": f"{it_sdet}{it_subj} {it_verb} {it_odet} {it_ob}", #using sdet instead of odet for subj, because it's a word starting w a vowel, and therefore the det is not gendered
                        "en": f"{sdet} {subj} {verb} {odet} {ob}",
                        "en-r-1-subordinate": f"{pSubj} {pVerb} that the {subj} {verb} the {ob}",
                        "en-r-2-passive": f"{odet} {ob} is {passiveSeed[verb]} by {sdet} {subj}",
                        "en-u-1-negation": f"{sdet} {subj} {verb} {odet} doesn't {ob}",
                        "en-u-2-inversion": " ".join(f"{sdet} {subj} {verb} {odet} {ob}".split(" ")[::-1]),
                        "en-u-3-qsubordinate": f"{pSubj} {pVerb} that does the {subj} {verb} the {ob}?"
                    }])])
                elif (it_sdet == "l'" and it_odet == "l'"):
                    df = pd.concat([df, pd.DataFrame.from_dict([{
                        "ita": f"{it_sdet}{it_subj} {it_verb} {it_odet}{it_ob}",
                        "ita-r-1-null_subject": f"{it_verb} {it_odet}{it_ob}",
                        "ita-r-2-subordinate": f"{it_pSubj} {it_pVerb} che {it_sdet} {it_subj} {it_verb} {it_odet}{it_ob}",
                        "ita-r-3-passive":ita_passive_sentence,
                        "ita-u-1-negation": f"{it_sdet}{it_subj} {it_verb} il no {it_ob}",
                        "ita-u-2-invert": " ".join( f"{it_sdet}{it_subj} {it_verb} {it_odet}{it_ob}".split(" ")[::-1]),
                        "ita-u-3-gender": f"{it_odet}{it_subj} {it_verb} {it_odet}{it_ob}",
                        "en": f"{sdet} {subj} {verb} {odet} {ob}",
                        "en-r-1-subordinate": f"{pSubj} {pVerb} that the {subj} {verb} the {ob}",
                        "en-r-2-passive": f"{odet} {ob} is {passiveSeed[verb]} by {sdet} {subj}",
                        "en-u-1-negation": f"{sdet} {subj} {verb} {odet} doesn't {ob}",
                        "en-u-2-inversion": " ".join(f"{sdet} {subj} {verb} {odet} {ob}".split(" ")[::-1]),
                        "en-u-3-qsubordinate": f"{pSubj} {pVerb} that does the {subj} {verb} the {ob}?"
                    }])])
                else:
                    df = pd.concat([df, pd.DataFrame.from_dict([{
                        "ita": f"{it_sdet} {it_subj} {it_verb} {it_odet}{it_ob}",
                        "ita-r-1-null_subject": f"{it_verb} {it_odet}{it_ob}",
                        "ita-r-2-subordinate": f"{it_pSubj} {it_pVerb} che {it_sdet}{it_subj} {it_verb} {it_odet}{it_ob}",
                        "ita-r-3-passive":ita_passive_sentence,
                        "ita-u-1-negation": f"{it_sdet} {it_subj} {it_verb} {it_odet} no {it_ob}",
                        "ita-u-2-invert": " ".join(f"{it_sdet} {it_subj} {it_verb} {it_odet}{it_ob}".split(" ")[::-1]),
                        "ita-u-3-gender": f"il {it_subj} {it_verb} {it_odet}{it_ob}",
                        "en": f"{sdet} {subj} {verb} {odet} {ob}",
                        "en-r-1-subordinate": f"{pSubj} {pVerb} that the {subj} {verb} the {ob}",
                        "en-r-2-passive": f"{odet} {ob} is {passiveSeed[verb]} by {sdet} {subj}",
                        "en-u-1-negation": f"{sdet} {subj} {verb} {odet} doesn't {ob}",
                        "en-u-2-inversion": " ".join(f"{sdet} {subj} {verb} {odet} {ob}".split(" ")[::-1]),
                        "en-u-3-qsubordinate": f"{pSubj} {pVerb} that does the {subj} {verb} the {ob}?"
                    }])])



pluralSeed = [ { 'verb' : ['carry', 'hold', 'take', 'bring'],  'subject': ['dogs', 'cats', 'men', 'women', 'teachers', 'girls', 'boys'], 'object': ['fish', 'mouse', 'bottle', 'newspaper', 'chalk', 'box', 'cap', 'bulb', 'cup', 'toy']},

{ 'verb': ['climb'], 'subject': ['dogs', 'cats', 'men', 'women', 'teachers', 'girls', 'boys'], 'object': ['staircase', 'rock', 'hill', 'mountain', 'roof', 'tree'] },

{ 'verb': ['eat'], 'subject' : ['dogs', 'cats', 'men', 'women', 'teachers', 'girls', 'boys'], 'object': ['biscuit', 'fish', 'banana', 'pear', 'meal', 'fruit', 'cucumber', 'pizza' ] },
{'verb': ['read'], 'subject' : ['poets', 'authors', 'writers', 'speakers', 'teachers', 'girls', 'boys'], 'object': ['book', 'poem', 'story', 'chapter']} ]

it_pluralSeed = [ { 'verb' : ['portano', 'tengono', 'prendono', 'portano'],  'subject': ['cani', 'gatti', 'uomini', 'donne', 'insegnanti', 'ragazze', 'ragazzi'], 'object': ['pesce', 'topo', 'bottiglia', 'giornale', 'gesso', 'scatola', 'cappello', 'lampadina', 'tazza', 'giocattolo']},

{ 'verb': ['salgono'], 'subject': ['cani', 'gatti', 'uomini', 'donne', 'insegnanti', 'ragazze', 'ragazzi'], 'object': ['scala', 'roccia', 'collina', 'montagna', 'tetto', 'albero'] },

{ 'verb': ['mangiano'], 'subject' : ['cani', 'gatti', 'uomini', 'donne', 'insegnanti', 'ragazze', 'ragazzi'], 'object': ['biscotto', 'pesce', 'banana', 'pera', 'pasto', 'frutta', 'cetriolo', 'pizza' ] },
{'verb': ['leggono'], 'subject' : ['poeti', 'autori', 'scrittori', 'oratori', 'insegnanti', 'ragazze', 'ragazzi'], 'object': ['libro', 'poema', 'storia', 'capitolo']} ]

it_subordinateSeed = [ { 'verb' : ['vede', 'dice', 'osserva', 'sa', 'afferma'],  'subject': ['Sheela', 'Leela', 'Maria', 'Gomu', 'John', 'Tom', 'Harry'], }]

for oidx, obj in enumerate(pluralSeed):
    for sidx, subj in enumerate(obj['subject']):
        for obidx, ob in enumerate(obj['object']):
            for vidx, verb in enumerate(obj['verb']):
                pSubj = random.choice(subordinateSeed[0]['subject'])
                pVerb = random.choice(subordinateSeed[0]['verb'])
                odet = random.choice(['the', 'a'])

                it_subj = it_pluralSeed[oidx]['subject'][sidx]
                it_sdet = it_genderDict[it_subj][0]
                it_ob = it_pluralSeed[oidx]['object'][obidx]
                it_odet = it_genderDict[it_ob][0]

                it_verb = it_pluralSeed[oidx]['verb'][vidx]

                it_pSubj = random.choice(it_subordinateSeed[0]['subject'])
                it_pVerb = random.choice(it_subordinateSeed[0]['verb'])

                temp_sdet = it_sdet
                if it_sdet == 'i':
                    temp_sdet = 'il'
                    ita_passive_from = 'dai'
                if it_sdet == 'le':
                    temp_sdet = 'la'
                    ita_passive_from = 'dalle'
                elif it_sdet == "gli":
                    temp_sdet = 'il'
                    ita_passive_from = "dagli'"
                ita_passive_sentence = f"{it_odet} {it_ob} {it_passiveSeed[it_verb][temp_sdet]} {ita_passive_from} {it_subj}"

                if it_odet == 'l':
                    temp_odet = 'il'
                else:
                    temp_odet = it_odet

                df = pd.concat([df, pd.DataFrame.from_dict([{
                    "ita": f"{it_sdet} {it_subj} {it_verb} {it_odet} {it_ob}",
                    "ita-r-1-null_subject": f"{it_verb} {it_odet} {it_ob}",
                    "ita-r-2-subordinate": f"{it_pSubj} {it_pVerb} che {it_sdet} {it_subj} {it_verb} {it_odet} {it_ob}",
                    "ita-r-3-passive": ita_passive_sentence,
                    "ita-u-1-negation": f"{it_sdet} {it_subj} {it_verb} {it_odet} no {it_ob}",
                    "ita-u-2-invert": " ".join(f"{it_sdet} {it_subj} {it_verb} {it_odet} {it_ob}".split(" ")[::-1]),
                    "ita-u-3-gender": f"{temp_odet} {it_subj} {it_verb} {it_odet} {it_ob}",
                    "en": f"the {subj} {verb} {odet} {ob}",
                    "en-r-1-subordinate": f"{pSubj} {pVerb} that the {subj} {verb} the {ob}",
                    "en-r-2-passive": f"{odet} {ob} is {passiveSeed[verb]} by the {subj}",
                    "en-u-1-negation": f"{sdet} {subj} {verb} {odet} doesn't {ob}",
                    "en-u-2-inversion": " ".join(f"{sdet} {subj} {verb} {odet} {ob}".split(" ")[::-1]),
                    "en-u-3-qsubordinate": f"{pSubj} {pVerb} that does the {subj} {verb} the {ob}?"
                }])])
                if (f"{odet} {ob} is {passiveSeed[verb]} by the {subj}" == None):
                    print(f"{odet} {ob} is {passiveSeed[verb]} by the {subj}", odet, ob, passiveSeed[verb], subj)

df.reset_index()

## Italian sentenc

df['it'] = df.progress_apply(lambda row: " ".join([genderDict[row['en'].split(" ")[1]]] + row['en'].split(" ")[1:3] + [genderDict[row['en'].split(" ")[4]]] + [row['en'].split(" ")[-1]]), axis=1)

## IT Real grammar 1 (Null Subject parameter

df['it-r-1-null_subject'] = df.progress_apply(lambda row: " ".join(row['it'].split(" ")[2:]), axis=1)

## IT Real Grammar 2 (Passive construction

df['it-r-2-passive'] =  df.progress_apply(lambda row: " ".join([genderDict[row['en-r-2-passive'].split(" ")[1]]] + row['en-r-2-passive'].split(" ")[1:-2] + [genderDict[row['en-r-2-passive'].split(" ")[-1]]] + [row['en-r-2-passive'].split(" ")[-1]]), axis=1)

## IT Real Grammar 3 (Subordinate construction

df['it-r-3-subordinate'] =  df.progress_apply(lambda row: " ".join(row['en-r-1-subordinate'].split(" ")[0:3] + row['it'].split(" ")), axis=1)

## IT Unreal Grammar 1: Add a negation after the 3rd word in the nullified subject sentenc

df['it-u-1-negation'] = df.progress_apply(lambda row: " ".join(row['it'].split(" ")[:4] + [ "no" ] + row['en'].split(" ")[4:]), axis=1)

## IT Unreal Grammar 2: Invert italian sentenc

df['it-u-2-invert'] = df.progress_apply(lambda row: " ".join(row['it'].split(" ")[::-1]), axis=1)

## IT Unreal Grammar 3: Same gender for subject and objec

df['it-u-3-gender'] = df.progress_apply(lambda row: " ".join(row['it'].split(" ")[:3] + [row['it'].split(" ")[0]] + [row['it'].split(" ")[-1]]), axis=1)

## JP real grammar 1 (Wa after subj, o after obj, verb

df['jp-r-1-sov'] = df.progress_apply(lambda row: " ".join(row["en"].split(" ")[:2]) + " wa " + " ".join(row["en"].split(" ")[-2:]) + " o " + row["en"].split(" ")[2], axis=1)

## JP real grammar 2 (Passive construction

df['jp-r-2-passive'] = df.progress_apply(lambda row: " ".join(row["en"].split(" ")[3:5]) + " wa " + " ".join(row["en"].split(" ")[:2]) + " ni " + infinitive[row["en"].split(" ")[2]] + " reru", axis=1)

## JP real grammar 3 (Subordinate construction

df['jp-r-3-subordinate'] = df.progress_apply(lambda row: " ".join([row["en-r-1-subordinate"].split(" ")[0]] + ["wa"] + row["en-r-1-subordinate"].split(" ")[3:5] + ["ga"] + row["en-r-1-subordinate"].split(" ")[-2:]  + ["o"] + [row["en-r-1-subordinate"].split(" ")[5]] + ["to"] + [row["en-r-1-subordinate"].split(" ")[1]]), axis=1)

## JP - Unreal grammar 1:Add a negation at the end of the object in the real-jp-1 sentenc

df['jp-u-1-negation'] = df.progress_apply(lambda row: " ".join(row['jp-r-1-sov'].split(" ")[:3]) + " no " + " ".join(row['jp-r-1-sov'].split(" ")[3:]), axis=1)

## JP - Unreal grammar 2: Invert jp-real-1 sentenc

df['jp-u-2-invert'] = df.progress_apply(lambda row: " ".join(row['jp-r-1-sov'].split(" ")[::-1]), axis=1)

## JP - Unreal grammar add a after o + past tens

df['jp-u-3-past-tense'] = df.progress_apply(lambda row: " ".join(row['jp-r-1-sov'].split(" ")[:-2] + [' o-ta '] + [infinitive[row['jp-r-1-sov'].split(" ")[-1]]]), axis=1)

## Non-grammatical sentence


def swap_words(sentence, col):
    sentence = sentence.split(" ")
    numWords = len(sentence)
    toSwap = [0,1]
    toSwapWords = set([sentence[toSwap[0]],sentence[toSwap[1]]])
    swap1 = sentence[toSwap[0]]
    swap2 = sentence[toSwap[1]]
    sentence[toSwap[0]] = swap2
    sentence[toSwap[1]] = swap1
    return " ".join(sentence)

for col in list(df.columns):
    print(' Now processing.... ', col)
    df[f'ng-{col}'] = df.progress_apply(lambda row: swap_words(row[col], col), axis=1)


gCols = [col for col in df.columns if not 'ng' in col]
ngCols = [col for col in df.columns if 'ng' in col]


df.to_csv('ngs.csv', index=False)
