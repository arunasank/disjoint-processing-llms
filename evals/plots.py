import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
import torch
from matplotlib import patches
from tqdm import tqdm
import scipy
import statistics
from matplotlib.ticker import MaxNLocator
import ast
import scipy.stats as stats
from scipy.stats import chi2_contingency
from scipy.stats import mannwhitneyu
from tabulate import tabulate
from scipy.stats import combine_pvalues
import glob
import os

class Constants:
    def __init__(self):
        self.conventional_names = {'en-r-1': 'EN Declarative (H)',
        'en-r-2-subordinate': 'EN Subordinate (H)',
        'en-r-3-passive': 'EN Passive (H)',
        'en-u-1-negation': 'EN Negation (L)',
        'en-u-2-inversion':'EN Inversion (L)',
        'en-u-3-wh':'EN Wh- word (L)',
        'ita-r-1':'IT Declarative (H)',
        'ita-r-2-subordinate':'IT Subordinate (H)',
        'ita-r-3-passive':'IT Passive (H)',
        'ita-u-1-negation':'IT Negation (L)',
        'ita-u-2-inversion':'IT Inversion (L)',
        'ita-u-3-gender':'IT gender agreement (L)',
        'jap-r-1':'JP Declarative (H)',
        'jap-r-2-subordinate':'JP Subordinate (H)',
        'jap-r-3-passive':'JP Passive (H)',
        'jap-u-1-negation':'JP Negation(L)',
        'jap-u-2-inversion':'JP Inversion (L)',
        'jap-u-3-past-tense':'JP past-tense (L)',
        }

        self.nonce_names = new_names = {'en_S-r-1': 'EN Declarative (H)',
        'en_S-r-2-subordinate': 'EN Subordinate (H)',
        'en_S-r-3-passive': 'EN Passive (H)',
        'en_S-u-1-negation': 'EN Negation (L)',
        'en_S-u-2-inversion':'EN Inversion (L)',
        'en_S-u-3-wh':'EN Wh- word (L)',
        'ita_S-r-1':'IT Declarative (H)',
        'ita_S-r-2-subordinate':'IT Subordinate (H)',
        'ita_S-r-3-passive':'IT Passive (H)',
        'ita_S-u-1-negation':'IT Negation (L)',
        'ita_S-u-2-inversion':'IT Inversion (L)',
        'ita_S-u-3-gender':'IT gender agreement (L)',
        'jap_S-r-1':'JP Declarative (H)',
        'jap_S-r-2-subordinate':'JP Subordinate (H)',
        'jap_S-r-3-passive':'JP Passive (H)',
        'jap_S-u-1-negation':'JP Negation(L)',
        'jap_S-u-2-inversion':'JP Inversion (L)',
        'jap_S-u-3-past-tense':'JP past-tense (L)',
        }

        self.real_color = '#007FFF'
        self.unreal_color = '#E55451'
        self.mixed_color = '#6300A9'
        
class Config:
    def __init__(self, model_name, token_type):
        self.constants = Constants()
        self.model_name = model_name
        self.token_type = token_type
        self.grammar_names = []
        if token_type == 'nonce':
            model_dirs = ['test-10']
            self.grammar_names = self.constants.nonce_names
        elif token_type == 'conventional':
            model_dirs = ['10-seed-10']
            self.grammar_names = self.constants.conventional_names
        else:
            assert False, f"Enter the appropriate token type {token_type}"
        self.model_dirs_exp_1 = [f"/mnt/align4_drive/arunas/broca/{model_name}/experiments/{m_dir}/" for m_dir in model_dirs]
        self.model_dirs_exp_2 = { "attn": 
            [f"/mnt/align4_drive/arunas/broca/{model_name}/atp/patches/attn/{m_dir}" for m_dir in model_dirs],
            "mlp": [f"/mnt/align4_drive/arunas/broca/{model_name}/atp/patches/mlp/{m_dir}" for m_dir in model_dirs]
        }
        self.op_dirs = self.make_op_dirs()
        
    def make_op_dirs(self):
        op = f'/mnt/align4_drive/arunas/broca/plots/expt'
        op_dirs = {}
        for i in range(1,4):
            os.makedirs(f'{op}{i}', exist_ok=True)
            op_dirs[f'expt{i}'] = f'{op}{i}'
        return op_dirs
        
        
    def get_files(self, m_dir):
            dir_list = glob.glob(f'{m_dir}/*-accuracy.csv')
            acc = pd.DataFrame(columns=['acc', 'language', 'grammar', 'real'])
            assert len(dir_list) != 0
            for d in dir_list:
                d = pd.read_csv(d).head(1)
                d['grammar'] = d['lang']
                d['real'] = d['grammar'].apply(lambda x: 0 if '-u-' in x else 1)
                if self.token_type == 'conventional':
                    d['language'] = d['grammar'].apply(lambda x: x.split('-')[0].upper())
                elif self.token_type == 'nonce':
                    d['language'] = d['grammar'].apply(lambda x: x.split('_S')[0].upper())
                acc = pd.concat([acc, pd.DataFrame.from_dict({'acc': d['acc'], 'language': d['language'], 'grammar': d['grammar'], 'real': d['real']})])
            acc = acc.sort_values(by='grammar', key=lambda x: x.str.lower())
            
            return acc
    
    
class Expt1:
    def __init__(self, config):
        self.get_files = config.get_files
        self.model_dirs = config.model_dirs_exp_1
        self.model_name = config.model_name
        self.token_type = config.token_type
        self.grammar_names = config.grammar_names
        self.constants = config.constants
        self.op_dir = config.op_dirs['expt1']
    def create_plot(self):
        for m_dir in self.model_dirs:
            self.model_df = self.create_df()
            self.create_plot_lang(m_dir)
            self.create_plot_grammar(m_dir)
        
    def create_plot_grammar(self, m_dir):
        plt.rcParams.update({'font.size': 21})
        
        lang = self.model_df['language']
        colors = []
        for _, row in self.model_df.iterrows():
            colors.append(self.constants.real_color if row['real'] == 1 else self.constants.unreal_color)
            
        plt.figure(figsize=(12, 15))
        accuracies = self.model_df['acc']
        plt.bar(sorted(self.grammar_names.values()), accuracies, color=colors)
        plt.ylabel('Accuracy')
        plt.ylim(0,1)

        # Creating legend handles
        real_patch = plt.Line2D([0], [0], color=self.constants.real_color, lw=4, label='Hierarchical')
        unreal_patch = plt.Line2D([0], [0], color=self.constants.unreal_color, lw=4, label='Linear')

        plt.xticks(rotation=90)
        plt.legend(handles=[real_patch, unreal_patch], ncol=2)
        plt.tight_layout()
        plt.savefig(f'{m_dir}/{self.model_name}-grammars.png', dpi=300)
        plt.show()

    def create_plot_lang(self, m_dir):
        plt.rcParams.update({'font.size': 48})
        lang = self.model_df['language'].unique()
        r_data = []
        u_data = []
        for l in lang:
            r_data.append(list(self.model_df[(self.model_df['real'] == 1) & (self.model_df['language'] == l)]['acc']))
            u_data.append(list(self.model_df[(self.model_df['real'] == 0) & (self.model_df['language'] == l)]['acc']))
        
        r_means = [statistics.mean(rd) for rd in r_data]
        u_means = [statistics.mean(ud) for ud in u_data]
        r_stderr = [np.std(rd)/np.sqrt(len(rd)) for rd in r_data]
        u_stderr = [np.std(ud)/np.sqrt(len(ud)) for ud in u_data]
        
        n = len(lang)
        
        fig, ax = plt.subplots(figsize=(20,15))
    
        bar_width = 0.35
        index = np.arange(n)

        ax.bar(index, r_means, bar_width, label='Hierarchical', color=[self.constants.real_color])
        ax.bar(index + bar_width, u_means, bar_width, label='Linear', color=[self.constants.unreal_color])
        plt.grid(True, which='major', axis='y', linestyle='--', linewidth=1.5, color='gray', alpha=1)
        plt.errorbar(index, y=r_means, yerr=r_stderr, fmt='none', c='black', capsize=5)
        plt.errorbar(index + bar_width, y=u_means, yerr=u_stderr, fmt='none', c='black', capsize=5)
        ax.set_ylabel('Mean Accuracy')
        ax.set_ylim(0,1)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_xticks(index + bar_width/2)
        ax.set_xticklabels(lang)
        ax.legend(frameon=False, ncol=2)
        plt.tight_layout()
        fig.savefig(f'{self.op_dir}/{self.model_name}-lang.png', dpi=300)   
    
    def create_df(self):
        model_df = []
        for m_dir in self.model_dirs:
            model_df.append(self.get_files(m_dir))
        model_df = pd.concat(model_df)
        model_df = model_df.reset_index()
        
        return model_df
        
class Expt2:
    def __init__(self, config):
        self.get_files = config.get_files
        self.model_dirs = config.model_dirs_exp_2
        self.model_name = config.model_name
        self.token_type = config.token_type
        self.grammar_names = config.grammar_names
        self.attn_pickle_dirs = config.model_dirs_exp_2['attn']
        self.mlp_pickle_dirs = config.model_dirs_exp_2['mlp']
        self.op_dir = config.op_dir['expt2']
        
    def create_plot(self):
        self.conf_matrix_plot(self.model_name, 'attn', 0.01, self.attn_pickle_dirs)
        self.conf_matrix_plot(self.model_name, 'mlp', 0.01, self.mlp_pickle_dirs)
        
    def conf_matrix_plot(self, model, compPath, topK, dirs):
        def conf_matrix_individual(model, topK, pkl_dir):
            global component
            columns = self.grammar_names.keys()
            reals = [col for col in sorted(columns) if not '-u-' in col]
            unreals = [col for col in sorted(columns) if '-u-' in col]
            component = pd.DataFrame(columns=np.arange(0,65530), index=sorted(columns))
            component_values = pd.DataFrame(columns=np.arange(0,65530), index=sorted(columns))
            index = 0
            
            for col in columns:
                try:
                    print(f'{pkl_dir}/{col}.pkl')
                    with open(f'{pkl_dir}/{col}.pkl', 'rb') as f:
                        component_cache = pickle.load(f)
                        component_cache = component_cache.cpu()
                        flattened_effects_cache = component_cache.view(-1)
                        top_neurons = flattened_effects_cache.topk(k=int(topK * flattened_effects_cache.shape[-1]))
                        two_d_indices = torch.cat((((top_neurons[1] // component_cache.shape[1]).unsqueeze(1)), ((top_neurons[1] % component_cache.shape[1]).unsqueeze(1))), dim=1)            
                        df = pd.DataFrame(two_d_indices, columns=['layer', 'neuron'])
                        two_d_indices_1 = torch.cat((((top_neurons[1] // component_cache.shape[1]).unsqueeze(1)), ((top_neurons[1] % component_cache.shape[1]).unsqueeze(1)), top_neurons[0].unsqueeze(1)), dim=1)            
                        df_1 = pd.DataFrame(two_d_indices_1, columns=['layer', 'neuron', 'value'])

                        for idx, row in df.iterrows():
                            component.loc[(component.index==col),idx] = ",".join([str(row['layer']), str(row['neuron'])])
                            index = idx
                        for idx, row in df_1.iterrows():
                            component_values.loc[(component_values.index==col),idx] = ",".join([str(row['layer']), str(row['neuron']), str(row['value'])])
                            
                except:
                    print(col)
            print(index)
            print('Gathered pickles')
            component = component[list(component.columns)[:index]]
            fig_column_order = [col for col in component.index if not '-u-' in col] + [col for col in component.index if '-u-' in col]
            component_overlap = pd.DataFrame(columns=fig_column_order, index=fig_column_order)

                
            for col in tqdm(fig_column_order):
                for col1 in fig_column_order:
                    component_overlap.loc[(component_overlap.index == col), col1] = \
                        len(set(component.loc[col]).intersection(set(component.loc[col1])))/len(component.columns)
                component_overlap[col] = component_overlap[col].astype(float)

            print('NUM NEURONS ', len(component.columns))
            return component_overlap, component_values, fig_column_order
        
        overall_comp_path = []
        overall_comp_values = []
        fig_column_order  = 0
        for _dir in dirs:
            component_overlap, component_values, fig_column_order = conf_matrix_individual(model, topK, _dir)
            overall_comp_path.append(component_overlap)
            overall_comp_values.append(component_values)
            
        average_df = pd.DataFrame(columns=fig_column_order, index=fig_column_order)
        
        for row in fig_column_order:
            for col in fig_column_order:
                average_df.loc[(average_df.index == row), col] = float(np.mean([overall_comp_path[i].loc[(overall_comp_path[i].index == row), col] for i in range(len(overall_comp_path))]))
        
        average_df = average_df.apply(pd.to_numeric)
        # print(average_df)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(average_df, cmap='binary', aspect='auto', interpolation='nearest')
        ax = plt.gca()
        plt.xticks(range(len(fig_column_order)), labels=[self.grammar_names[col] for col in fig_column_order])
        plt.yticks(range(len(sorted(average_df.index))), labels=[self.grammar_names[col] for col in fig_column_order])
        plt.xticks(fontsize=12, rotation=-45, ha='left')  # Set x-axis tick label font size
        plt.yticks(fontsize=12)  # Set y-axis tick label font size

        cbar = plt.colorbar()
        cbar.set_label('Overlap %', fontsize=15)
        cbar.ax.tick_params(labelsize=15) 
        
        plt.title(f'{model.title()} {compPath.title()}', fontsize=15)
        plt.tight_layout()

        # -------------------------------- Draw lines -------------------------------------------------
        plt.axvline(x=8.5, color='black')
        # plt.axvline(x=ux_tick[0], color=unreal_color)
        plt.axhline(y=8.5, color='black')
        # plt.axhline(y=ux_tick[0], color=unreal_color)
        plt.savefig(f'{self.op_dir}/{model}-{compPath}-overlap-{topK}.png')
        plt.show()
        return overall_comp_path, overall_comp_values
    
    
def main():
    MODEL_NAMES = ['gpt-2-xl', 'llama-2-7b', 'llama-3.1-8b', 'mistral-v0.3', 'qwen-2-0.5b', 'llama-3.1-70b', 'qwen-2-1.5b']
    TOKEN_TYPES = ['nonce', 'conventional']
    
    for model_name in MODEL_NAMES:
        for token_type in TOKEN_TYPES:
            try:
                print(f"########## Creating experiment 1 plots for {model_name} and {token_type} tokens dataset")
                config = Config(model_name, token_type)
                expt_1 = Expt1(config)
                expt_1.create_plot()
                
                print(f"########## Creating experiment 2 plots for {model_name} and {token_type} tokens dataset")
                expt_2 = Expt2(config)
                expt_2.create_plot()
            except:
                print(f"Failed to create plots for {model_name} and {token_type} tokens dataset")
                continue
            
if __name__ == "__main__":
    main()
    