import pandas as pd
import argparse

from transformers import data

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_file', type=str, help='path to the data file to check.')
args = parser.parse_args()

datafile = args.data_file

dataset = pd.read_csv(datafile)

for col in dataset.columns:
    print('Checking column:', col)
    for item in dataset[col]:
        assert '[' not in item, f"Found '[' in {item} in column {col}"
        assert ']' not in item, f"Found ']' in {item} in column {col}"
