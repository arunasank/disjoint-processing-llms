import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import torch
columns = ['sentence', 'subordinate-sentence', 'passive-sentence', 'it', 'it-r-1-null_subject', 'it-r-2-passive', 'it-r-3-subordinate', 'it-u-1-negation', 'it-u-2-invert', 'it-u-3-gender', 'jp-r-1-sov', 'jp-r-2-passive', 'jp-r-3-subordinate', 'jp-u-1-negation',    'jp-u-2-invert', 'jp-u-3-past-tense']
mlp = pd.DataFrame(columns=columns+ ['layer', 'neuron'])
mlp['neuron'] = np.arange(0,4096)
mlp[columns] = 0
mlp['layer'] = 31
for col in columns:
    with open(f'/home/gridsan/arunas/broca/mistral/mistral-attr-patch-scripts/mlp/{col}.pkl', 'rb') as f:
        x = torch.load(f,map_location=torch.device('cpu'))
        x = x.cpu()
        df = pd.DataFrame(x, columns=['layer', 'neuron'])
        for idx, row in df.iterrows():
            mlp.loc[(mlp['neuron'] == row['neuron']) & (mlp['layer'] == row['layer']), col] = 1
mlp = mlp[~(mlp[columns] == 0).all(axis=1)]
mlp = mlp[['sentence', 'subordinate-sentence', 'passive-sentence', 'it-r-2-passive', 'it-r-3-subordinate', 'jp-u-3-past-tense']]
styled_df = mlp.style.background_gradient(cmap='viridis')
styled_html = styled_df.to_html(style={'width': '40px', 'height': '60px'})
config_path = '/home/gridsan/arunas/broca/mistral/mistral-attr-patch-scripts'
imgkit.from_string(html, 'styled_table.png')
