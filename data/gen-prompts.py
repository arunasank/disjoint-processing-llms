import torch
from datasets import Dataset
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import random
import argparse

parser = argparse.ArgumentParser(description='Generate prompts for synthetic data')
parser.add_argument('--train_seed', type=int, default=42, help='Training demonstrations seed')
parser.add_argument('--num_dems', type=int, default=10, help='Number of demonstrations')
parser.add_argument('--test_seed', type=int, default=10, help='Test data seed')
parser.add_argument('--type', type=str, default='nonce', help='Type of token')
args = parser.parse_args()
GLOBAL_RANDOM_SEED = args.train_seed
TEST_DATA_SEED = args.test_seed
TOKEN_TYPE = args.type
torch.manual_seed(GLOBAL_RANDOM_SEED)
np.random.seed(GLOBAL_RANDOM_SEED)
random.seed(GLOBAL_RANDOM_SEED)
PREFIX = '/mnt/align4_drive/arunas/broca/data-gen'
NUM_DEMONSTRATIONS = 10
df = pd.read_csv(f'{PREFIX}/ngs-08-01-2024-synthetic-grammars-nonce.csv')
def get_g_cols():
    global df
    global TOKEN_TYPE
    g_cols = []
    if TOKEN_TYPE == "conventional":
        g_cols = sorted([col for col in df.columns \
            if not ('_S' in col) and not('ng-' in col) \
            and (not 'qsub' in col) and (not 'null_subject' in col) \
            and ('ita-' in col  or 'en-' in col or 'jap-' in col)
        ])
    elif TOKEN_TYPE == "nonce":
        g_cols = [col for col in sorted(df.columns) \
            if not ('ng-' in col) and '_S' in col]
    
    assert len(g_cols) != 0, f"g_cols cannot be empty {g_cols}"
    return g_cols
gCols = get_g_cols()

def parse_answer(text):
    answers = []
    for t in text:
        ans = t.split("A:")[-1].strip()
        answers.append(ans)
    return answers

class WordIdentifierMapper:
    def __init__(self):
        self.mapping = {}
        self.counter = 0

    def generate_identifier(self):
        """Generate the next unique identifier."""
        identifier = ""
        temp_counter = self.counter
        while temp_counter >= 0:
            identifier = chr((temp_counter % 26) + ord('A')) + identifier
            temp_counter = temp_counter // 26 - 1
        self.counter += 1
        return identifier

    def map_sentence(self, sentence):
        """Map each word in the sentence using the global mapping."""
        words = sentence.split()
        mapped_sentence = []
        for word in words:
            if word not in self.mapping:
                self.mapping[word] = self.generate_identifier()
            mapped_sentence.append(self.mapping[word])
        return " ".join(mapped_sentence)

    def get_mapping(self):
        """Return the current global mapping."""
        return self.mapping

mapper = WordIdentifierMapper()

def construct_prompt(train_dataset, dem_indices, label_indices, num_demonstrations):
    assert num_demonstrations > 0
    prompt = ''
    exemplars = []
    train_examples = train_dataset.select(dem_indices)

    for exemplar_num in range(num_demonstrations):
        train_example = train_examples[exemplar_num]
        use_bad_sentence = label_indices[exemplar_num]
        exemplar = "Q: Is this sentence grammatical? Yes or No: "
        if use_bad_sentence:
            exemplar += mapper.map_sentence(train_example["ng-" + col])
            exemplar += "\nA: No"
        else:
            exemplar += mapper.map_sentence(train_example[col])
            exemplar += "\nA: Yes"
        exemplars.append(exemplar)    
        exemplar += "\n\n"
        prompt += exemplar
    return prompt, exemplars

for col in gCols:
    datasets = {}
    
    if TOKEN_TYPE == 'conventional':
        datasets[col] = Dataset.from_pandas(pd.DataFrame(df[[col, 'ng-' + col]].copy())).train_test_split(test_size=0.5, seed=TEST_DATA_SEED)
    elif TOKEN_TYPE == 'nonce':
        datasets[col] = Dataset.from_pandas(pd.DataFrame(df[[col, 'ng-' + col]].copy())).train_test_split(test_size=0.001, seed=TEST_DATA_SEED)
    def get_master_prompt(lang, idx):
        if 'en_S' == "".join(lang[:4]):
            intro = ["Here are English sentences with nonce words that either follow or break a grammar rule. Each sentence is labeled 'Yes' if it follows the rule and 'No' if it doesn't. Label the final sentence as 'Yes' or 'No' based on whether it follows the same rule.",
                     "",
                     "Here are some examples of sentences that follow a rule. Label the last sentence 'Yes' if it follows the same rule and 'No' if it doesn't.",
                ]
            return f"{intro[idx]}"
        
        elif 'en' == "".join(lang[:2]):
            intro = "Here are English sentences that either follow or break a grammar rule. Each sentence is labeled 'Yes' if it follows the rule and 'No' if it doesn't. Label the final sentence as 'Yes' or 'No' based on whether it follows the same rule."
            return f"{intro}"

        elif 'ita_S' == "".join(lang[:2]):
            intro = ["Here are Italian sentences with nonce words that either follow or break a grammar rule. Each sentence is labeled 'Yes' if it follows the rule and 'No' if it doesn't. Label the final sentence as 'Yes' or 'No' based on whether it follows the same rule.",
                     "",
                     "Here are some examples of sentences that follow a rule. Label the last sentence 'Yes' if it follows the same rule and 'No' if it doesn't.",
                ]
            return f"{intro[idx]}"
        elif 'ita' == "".join(lang[:3]):
            intro = "Here are Italian sentences that either follow or break a grammar rule. Each sentence is labeled 'Yes' if it follows the rule and 'No' if it doesn't. Label the final sentence as 'Yes' or 'No' based on whether it follows the same rule."
            return f"{intro}"
                
        elif 'jap_S' == "".join(lang[:4]):
            intro = ["Here are Japanese sentences with nonce words that either follow or break a grammar rule. Each sentence is labeled 'Yes' if it follows the rule and 'No' if it doesn't. Label the final sentence as 'Yes' or 'No' based on whether it follows the same rule.",
                     "",
                     "Here are some examples of sentences that follow a rule. Label the last sentence 'Yes' if it follows the same rule and 'No' if it doesn't.",
                ]
            return f"{intro[idx]}"
        
        elif 'jap' == "".join(lang[:3]):
            intro = "Here are Japanese sentences that either follow or break a grammar rule. Each sentence is labeled 'Yes' if it follows the rule and 'No' if it doesn't. Label the final sentence as 'Yes' or 'No' based on whether it follows the same rule."
            return f"{intro}"

    def generate_nonce_test_sentence(sentences, input_label, col, test_num, TEST_SENTENCES):
        condition_dict = {
            "en_S-r-1": {"any sentence": [[0, 1]], "sentence with same input label": [[-2, -1]]},
            "en_S-r-2-subordinate": {"any sentence": [[3, 4]], "sentence with same input label": [[-2, -1]]},
            "en_S-r-3-passive": {"any sentence": [[0, 1]], "sentence with same input label": [[-2, -1]]},
            "en_S-u-1-negation": {"any sentence": [[0, 1]], "sentence with same input label": [[-3, -2, -1]]},
            "en_S-u-2-inversion": {"any sentence": [[0, 1]], "sentence with same input label": [[-2, -1]]},
            "en_S-u-3-wh": {"any sentence": [[1, 2]], "sentence with same input label": [[-3, -2, -1]]},
            "ita_S-r-1": {"any sentence": [[0, 1]], "sentence with same input label": [[-2, -1]]},
            "ita_S-r-2-subordinate": {"any sentence": [[3, 4]], "sentence with same input label": [[-2, -1]]},
            "ita_S-r-3-passive": {"any sentence": [[0, 1]], "sentence with same input label": [[-2, -1]]},
            "ita_S-u-1-negation": {"any sentence": [[0, 1]], "sentence with same input label": [[-3, -2, -1]]},
            "ita_S-u-2-inversion": {"any sentence": [[0, 1]], "sentence with same input label": [[-2, -1]]},
            "ita_S-u-3-gender": {"any sentence": [], "sentence with same input label": [[0, 1, -2, -1]]},
            "jap_S-r-1": {"any sentence": [[0, 1], [3, 4]], "sentence with same input label": [[-2, -1]]},
            "jap_S-r-2-subordinate": {"any sentence": [[2, 3], [5, 6]], "sentence with same input label": [[-2, -1]]},
            "jap_S-r-3-passive": {"any sentence": [[0, 1], [3, 4]], "sentence with same input label": [[-2, -1]]},
            "jap_S-u-1-negation": {"any sentence": [[0, 1], [3, 4]], "sentence with same input label": [[-3, -2, -1]]},
            "jap_S-u-2-inversion": {"any sentence": [[2, 3]], "sentence with same input label": [[-2, -1]]},
            "jap_S-u-3-past-tense": {"any sentence": [[0, 1], [3, 4]], "sentence with same input label": [[-3, -2, -1]]},
        }

        # Parse sentences into components starting after the specified prefix
        parsed_sentences = []
        prefix = "Q: Is this sentence grammatical? Yes or No: "
        
        for s in sentences:
            if prefix in s:
                main_part = s.split(prefix)[1].strip()
                words = main_part.split("\n")[0].split()
                label = main_part.split("\n")[-1].split(":")[-1].strip()
                parsed_sentences.append((words, label))
        
        # Get the condition for the specified col
        condition = condition_dict[col]
        any_sentence_positions = condition["any sentence"]
        same_label_positions = condition["sentence with same input label"]
        
        # Find the maximum number of words in any sentence
        assert len(set(len(s[0]) for s in parsed_sentences)) == 1, "All parsed sentences should have the same length"
        max_words = len(parsed_sentences[0][0])
        
        # Generate the test sentence
        test_sentence = [None] * max_words
        
        dont_use_these = []
        ctr = 0
        while ctr < len(parsed_sentences):
            # Fill positions from "any sentence"
            for pos_pair in any_sentence_positions:
                sentence = random.choice(parsed_sentences)[0]
                dont_use_these.append(sentence)
                for pos in pos_pair:
                    test_sentence[pos] = sentence[pos]
            
            # Fill positions from "sentence with same input label"
            filtered_sentences = [s[0] for s in parsed_sentences if s[1] == input_label]
            for pos_pair in same_label_positions:
                sentence = random.choice(filtered_sentences)
                # dont_use_these.append(sentence)
                for pos in pos_pair:
                    test_sentence[pos] = sentence[pos]
            
            use_these = [sentence for sentence in parsed_sentences if not sentence in dont_use_these]
            # Fill remaining positions randomly
            for i in range(max_words):
                if test_sentence[i] is None:
                    sentence = random.choice(use_these)[0]
                    test_sentence[i] = sentence[i % len(sentence)]
            
            assert not None in test_sentence, "All positions should be filled"
            # Ensure uniqueness
            test_sentence_string = " ".join(test_sentence)
            
            # Find the index of test_sentence_string in parsed_sentences
            index = next(
                (i for i, s in enumerate(parsed_sentences) if " ".join(s[0]) == test_sentence_string),
                None
            )
            if index is None and test_sentence_string not in TEST_SENTENCES:
                return test_sentence_string
            else:
                test_sentence = [None] * max_words
                ctr += 1
        
        # Add the label at the end
        print(f"Couldn't generate a test sentence for {col} at index {test_num}")
        return None

    for idx in range(1,2):
        print("### IDX ", idx)
        master_prompt = get_master_prompt(col, idx)
        train_dataset = datasets[col]['train']
        test_dataset = datasets[col]['test']
        np.random.seed(GLOBAL_RANDOM_SEED)
        
        if TOKEN_TYPE == 'conventional':
            DATASET_RANGE = len(test_dataset)
        elif TOKEN_TYPE == 'nonce':
            DATASET_RANGE = len(train_dataset)//2
        else:
            assert False, f"Invalid token type {TOKEN_TYPE}"
        
        dem_indices = [np.random.choice(len(train_dataset), NUM_DEMONSTRATIONS, replace=False).tolist() for _ in range(DATASET_RANGE)]

        label_indices = [ 5 *[1] + 5 *[0] for _ in range(DATASET_RANGE)]
        for l_i in label_indices:
            np.random.shuffle(l_i)
        print(col, dem_indices[0])
        print(col, label_indices[0])
        f = pd.DataFrame(columns=["prompt", "q", "gold"])
        np.random.seed(TEST_DATA_SEED)
        
        i = 0
        ALL_TEST_SENTENCES = []
        while i < DATASET_RANGE and i < len(train_dataset):
            testBadOrGood = np.random.choice(['ng-', ''])

            if testBadOrGood == 'ng-':
                fGold = 'No'
            else:
                fGold = 'Yes'

            prompt, exemplars = construct_prompt(train_dataset, dem_indices[i], label_indices[i], NUM_DEMONSTRATIONS)    
            fPrompt = prompt
            prompt += "Q: Is this sentence grammatical? Yes or No: "
            
            # Generate and append test sentence
            test_sentence = None
            if TOKEN_TYPE == 'conventional':
                test_sentence = test_dataset[i][testBadOrGood + col]
            elif TOKEN_TYPE == 'nonce':
                test_sentence = generate_nonce_test_sentence(exemplars, fGold, col, i, ALL_TEST_SENTENCES)
            if test_sentence is None:
                continue
            
            ALL_TEST_SENTENCES.append(test_sentence)
            
            prompt += test_sentence
            prompt += "\nA: "
            assert '[' not in test_sentence, f"Found '[' in {test_sentence}"  
            assert ']' not in test_sentence, f"Found ']' in {test_sentence}"
            f = pd.concat([f, pd.DataFrame([{'prompt': f"{master_prompt}\n\n{prompt}", 'q': test_sentence, 'gold': fGold}])]).reset_index(drop=True)
            i += 1
        if i >= len(train_dataset):
            assert False, f"Ran out of train data for {col}"
        # Evaluate
        if (TEST_DATA_SEED != 42):
            f.to_csv(f"{PREFIX}/prompts/jabberwocky/10-seed-{TEST_DATA_SEED}/{col}-gen-{idx}.csv", index=False)
        else:
            f.to_csv(f"{PREFIX}/prompts/jabberwocky/10/{col}-gen-{idx}.csv", index=False)
