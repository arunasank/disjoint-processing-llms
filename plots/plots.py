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

        self.nonce_names = {'en_S-r-1': 'EN Declarative (H)',
        'en_S-r-2-subordinate': 'EN Subordinate (H)',
        'en_S-r-3-passive': 'EN Passive (H)',
        'en_S-u-1-negation': 'EN Negation (L)',
        'en_S-u-2-inversion':'EN Inversion (L)',
        'en_S-u-3-wh':'EN Wh-word (L)',
        # 'ita_S-r-1':'IT Declarative (H)',
        # 'ita_S-r-2-subordinate':'IT Subordinate (H)',
        # 'ita_S-r-3-passive':'IT Passive (H)',
        # 'ita_S-u-1-negation':'IT Negation (L)',
        # 'ita_S-u-2-inversion':'IT Inversion (L)',
        # 'ita_S-u-3-gender':'IT gender agreement (L)',
        # 'jap_S-r-1':'JP Declarative (H)',
        # 'jap_S-r-2-subordinate':'JP Subordinate (H)',
        # 'jap_S-r-3-passive':'JP Passive (H)',
        # 'jap_S-u-1-negation':'JP Negation(L)',
        # 'jap_S-u-2-inversion':'JP Inversion (L)',
        # 'jap_S-u-3-past-tense':'JP past-tense (L)',
        }

        self.nonce_conv_names = {'en_S-r-1': 'EN NONCE Declarative (H)',
        'en_S-r-2-subordinate': 'EN NONCE Subordinate (H)',
        'en_S-r-3-passive': 'EN NONCE Passive (H)',
        'en_S-u-1-negation': 'EN NONCE Negation (L)',
        'en_S-u-2-inversion':'EN NONCE Inversion (L)',
        'en_S-u-3-wh':'EN NONCE Wh- word (L)',
        'en-r-1': 'EN Declarative (H)',
        'en-r-2-subordinate': 'EN Subordinate (H)',
        'en-r-3-passive': 'EN Passive (H)',
        'en-u-1-negation': 'EN Negation (L)',
        'en-u-2-inversion':'EN Inversion (L)',
        'en-u-3-wh':'EN Wh-word (L)'
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
            model_dirs = ['nonce-10']
            self.grammar_names = self.constants.nonce_names
        elif token_type == 'conventional':
            model_dirs = ['conventional-10']
            self.grammar_names = self.constants.conventional_names
        elif token_type == 'nonce_conv':
            model_dirs = ['conventional-10', 'nonce-10']
            self.grammar_names = self.constants.nonce_conv_names
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
        for i in [1,2,2.1,3]:
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
            self.create_plot_lang()
            self.create_plot_grammar()
        
    def create_plot_grammar(self):
        plt.rcParams.update({'font.size': 21})
        
        grammar = self.grammar_names.keys()
        colors = []
        for idx, row in self.model_df[self.model_df['grammar'].isin(grammar)].iterrows():
            colors.append(self.constants.real_color if row['real'] == 1 else self.constants.unreal_color)
            
        plt.figure(figsize=(12, 15))
        accuracies = self.model_df[self.model_df['grammar'].isin(grammar)]['acc']
        plt.bar(sorted(self.grammar_names.values()), accuracies, color=colors)
        plt.ylabel('Accuracy')
        plt.ylim(0,1)

        # Creating legend handles
        real_patch = plt.Line2D([0], [0], color=self.constants.real_color, lw=4, label='Hierarchical')
        unreal_patch = plt.Line2D([0], [0], color=self.constants.unreal_color, lw=4, label='Linear')

        plt.xticks(rotation=90)
        plt.legend(handles=[real_patch, unreal_patch], ncol=2)
        plt.tight_layout()
        plt.savefig(f'{self.op_dir}/{self.token_type}-{self.model_name}-grammars.png', dpi=300)
        plt.show()

    def create_plot_lang(self):
        plt.rcParams.update({'font.size': 48})
        grammars = self.grammar_names.keys()
        lang = self.model_df[self.model_df['grammar'].isin(grammars)]['language'].unique()
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
        fig.savefig(f'{self.op_dir}/{self.token_type}-{self.model_name}-lang.png', dpi=300)   
    
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
        self.op_dir = config.op_dirs['expt2']
        self.real_color = config.constants.real_color
        self.unreal_color = config.constants.unreal_color
        self.mixed_color = config.constants.mixed_color
        
    def create_plot(self):
        print('Confusion Matrix Plots')
        attn, attn_values = self.conf_matrix_plot(self.model_name, 'attn', 0.01, self.attn_pickle_dirs)
        mlp, mlp_values = self.conf_matrix_plot(self.model_name, 'mlp', 0.01, self.mlp_pickle_dirs)
        
        print('Mean plots')
        self.draw_overlap_bars_plot("attn", attn, self.model_name)
        self.draw_overlap_bars_plot("mlp", mlp, self.model_name)
        
        print(f"Language x Structure {self.model_name}")
        self.get_overlap_values_langxstructure(mlp_values[0], mlp, 'MLP')
        self.get_overlap_values_langxstructure(attn_values[0], attn, 'ATTN')

        print(f"Structure {self.model_name}")
        self.get_overlap_values_structure(mlp_values[0], mlp, 'MLP')
        self.get_overlap_values_structure(attn_values[0], attn, 'ATTN')
        
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
                # try:
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
                        
                # except:
                #     print(col)
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

            # print('NUM NEURONS ', len(component.columns))
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
        print(fig_column_order)
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
        plt.savefig(f'{self.op_dir}/{self.token_type}-{model}-{compPath}-overlap-{topK}.png')
        plt.show()
        return overall_comp_path, overall_comp_values
    
    def get_values_for_substr(self, substring, df):
        filtered_df = df[df.index.str.contains(substring)]
        layer_neuron_dict = {}
        for _, row in filtered_df.iterrows():
            items = [ r[1].split(',') for r in row.items() if type(r[1]) == str]
            for (layer, neuron, value) in items:
                layer = int(float(layer))
                neuron = int(float(neuron))
                value = float(value)
                if (layer, neuron) not in layer_neuron_dict:
                    layer_neuron_dict[f"{layer}-{neuron}"] = []
                layer_neuron_dict[f"{layer}-{neuron}"].append(value)
        # print(substring, len(layer_neuron_dict.keys()))
        return layer_neuron_dict

    def intersections(self, dict1, dict2):
        intersecting_keys = set(dict1.keys()).intersection(dict2.keys())
        # print(len(dict1.keys()), len(dict2.keys()), len(intersecting_keys))
        intersection = {}

        max_len = 0
        for key in intersecting_keys:
            list1 = dict1[key]
            list2 = dict2[key]
            combined_list = list1 + list2
            intersection[key] = combined_list
            max_len = max(len(intersection[key]), max_len)
        return intersection

    def get_overlap_values_langxstructure(self, comp_df, overlap, comp):
        reals, unreals, reals_unreals = self.get_overlaps(overlap)
        suffix = '_S-' if self.token_type == 'nonce' else '-'
        real_real_values = {key: [] for key in reals.keys() if len(reals[key]) > 0}
        unreal_unreal_values = {key: [] for key in reals.keys() if len(reals[key]) > 0}
        real_unreal_values = {key: [] for key in reals.keys() if len(reals[key]) > 0}

        for lang in real_real_values.keys():
            real_real_values[lang] = self.get_values_for_substr(f'{lang}{suffix}r-', comp_df)
            unreal_unreal_values[lang] = self.get_values_for_substr(f'{lang}{suffix}u-', comp_df)
            real_unreal_values[lang] = self.intersections(real_real_values[lang], unreal_unreal_values[lang])

        real_real_total = {key: [] for key in reals.keys() if len(reals[key]) > 0}
        unreal_unreal_total = {key: [] for key in reals.keys() if len(reals[key]) > 0}
        real_unreal_total_x_real = {key: [] for key in reals.keys() if len(reals[key]) > 0}
        real_unreal_total_x_unreal = {key: [] for key in reals.keys() if len(reals[key]) > 0}

        for lang in real_real_total:
            for key in real_real_values[lang].keys():
                for item in real_real_values[lang][key]:
                    real_real_total[lang].append(item)

            for key in unreal_unreal_values[lang].keys():
                for item in unreal_unreal_values[lang][key]:
                    unreal_unreal_total[lang].append(item)

            for key in real_unreal_values[lang].keys():
                list_len = len(real_unreal_values[lang][key])
                for item in real_unreal_values[lang][key][:list_len//2]:
                    real_unreal_total_x_real[lang].append(item)

            for key in real_unreal_values[lang].keys():
                list_len = len(real_unreal_values[lang][key])
                for item in real_unreal_values[lang][key][list_len//2:]:
                    real_unreal_total_x_unreal[lang].append(item)

        print(f"{'Component':<15} | {'LANG':<10} | {'HxL/HxH':<25} | {'HxL/LxL':<30}")
        print("-" * 85)
        for lang in real_real_values:
            real_unreal_real_ratio = sum(real_unreal_total_x_real[lang]) / sum(real_real_total[lang])
            real_unreal_unreal_ratio = sum(real_unreal_total_x_unreal[lang]) / sum(unreal_unreal_total[lang])

            print(f"{comp:<15} | {lang:<10} | {real_unreal_real_ratio: <25.2f} | {real_unreal_unreal_ratio: <30.2f}")
            
    def get_overlap_values_structure(self, comp_df, overlap, comp):

        real_values = []
        unreal_values = []


        real_values = self.get_values_for_substr(f'-r-', comp_df)
        unreal_values = self.get_values_for_substr(f'-u-', comp_df)
        real_unreal_values = self.intersections(real_values, unreal_values)

        # print(real_values)
        real_total = []
        unreal_total = []
        real_unreal_total_x_real = []
        real_unreal_total_x_unreal = []

        for key in real_values.keys():
            for item in real_values[key]:
                real_total.append(item)

        for key in unreal_values.keys():
            for item in unreal_values[key]:
                unreal_total.append(item)
        
        ctr = 0
        for key in real_unreal_values.keys():
            list_len = len(real_unreal_values[key])
            for item in real_unreal_values[key][:list_len//2]:
                real_unreal_total_x_real.append(item)
        for key in real_unreal_values.keys():
            list_len = len(real_unreal_values[key])
            for item in real_unreal_values[key][list_len//2:]:
                real_unreal_total_x_unreal.append(item)
                
        # Print table header
        print(f"{'Component':<15} | {'HxL/HxH':<25} | {'HxL/LxL':<30}")
        print("-" * 75)

        # Calculate and print ratios
        real_unreal_real_ratio = sum(real_unreal_total_x_real) / sum(real_total)
        real_unreal_unreal_ratio = sum(real_unreal_total_x_unreal) / sum(unreal_total)

        # Print formatted results
        print(f"{comp:<15} | {real_unreal_real_ratio:<25.2f} | {real_unreal_unreal_ratio:<25.2f}")

    def get_overlaps(self, conf_matrices):
        reals = { 'en': [], 'ita': [], 'jap': [] }
        unreals = { 'en': [], 'ita': [], 'jap': [] }
        reals_unreals = { 'en': [], 'ita': [], 'jap': [] }
        cols = conf_matrices[0].index
        for ind in cols:
            for col in cols:
                r_en = []
                u_en = []
                ru_en = []
                
                r_ita = []
                u_ita = []
                ru_ita = []
                
                r_jap = []
                u_jap = []
                ru_jap = []
                
                for conf_matrix in conf_matrices:
                    if (not '-u-' in col) and (not '-u-' in ind):
                        if ('en' in col[:2]) and ('en' in ind[:2]):
                            r_en.append(conf_matrix[(conf_matrix.index == ind)][col].item())
                        elif ('ita' in col[:3]) and ('ita' in ind[:3]):
                            r_ita.append(conf_matrix[(conf_matrix.index == ind)][col].item())
                        elif ('jap' in col[:3]) and ('jap' in ind[:3]):
                            r_jap.append(conf_matrix[(conf_matrix.index == ind)][col].item())
                    elif ('-u-' in col) and ('-u-' in ind):
                        if ('en' in col[:2]) and ('en' in ind[:2]):
                            u_en.append(conf_matrix[(conf_matrix.index == ind)][col].item())
                        elif ('ita' in col[:3]) and ('ita' in ind[:3]):
                            u_ita.append(conf_matrix[(conf_matrix.index == ind)][col].item())
                        elif ('jap' in col[:3]) and ('jap' in ind[:3]):
                            u_jap.append(conf_matrix[(conf_matrix.index == ind)][col].item())
                    elif ('-u-' in col) and ('-r-' in ind):
                        if ('en' in col[:2]) and ('en' in ind[:2]):
                            ru_en.append(conf_matrix[(conf_matrix.index == ind)][col].item())
                        elif ('ita' in col[:3]) and ('ita' in ind[:3]):
                            ru_ita.append(conf_matrix[(conf_matrix.index == ind)][col].item())
                        elif ('jap' in col[:3]) and ('jap' in ind[:3]):
                            ru_jap.append(conf_matrix[(conf_matrix.index == ind)][col].item())
                if (not '-u-' in col) and (not '-u-' in ind):
                    if ('en' in col[:2]) and ('en' in ind[:2]):
                        reals['en'].append(r_en)
                    elif ('ita' in col[:3]) and ('ita' in ind[:3]):
                        reals['ita'].append(r_ita)
                    elif ('jap' in col[:3]) and ('jap' in ind[:3]):
                        reals['jap'].append(r_jap)
                elif ('-u-' in col) and ('-u-' in ind):    
                    if ('en' in col[:2]) and ('en' in ind[:2]):
                        unreals['en'].append(u_en)
                    elif ('ita' in col[:3]) and ('ita' in ind[:3]):
                        unreals['ita'].append(u_ita)
                    elif ('jap' in col[:3]) and ('jap' in ind[:3]):
                        unreals['jap'].append(u_jap)
                elif  ('-u-' in col) and ('-r-' in ind):
                    if ('en' in col[:2]) and ('en' in ind[:2]):
                        reals_unreals['en'].append(ru_en)
                    elif ('ita' in col[:3]) and ('ita' in ind[:3]):
                        reals_unreals['ita'].append(ru_ita)
                    elif ('jap' in col[:3]) and ('jap' in ind[:3]):
                        reals_unreals['jap'].append(ru_jap)

        t_reals = [val for key in reals.keys() for val_list in reals[key] for val in val_list]
        t_unreals = [val for key in reals.keys() for val_list in unreals[key]  for val in val_list]
        t_reals_unreals = [val for key in reals.keys() for val_list in reals_unreals[key]  for val in val_list]
        print('H-H vs L-L', stats.mannwhitneyu(t_reals, t_unreals))
        print('H-H vs H-L', stats.mannwhitneyu(t_reals, t_reals_unreals))
        print('L-L vs H-L', stats.mannwhitneyu(t_unreals, t_reals_unreals))
        return reals, unreals, reals_unreals

    def draw_overlap_bars_plot(self, component, overlap, model_name):
        # Extract the categories and the values
        if component == 'MLP':
            print('MLP')
            reals, unreals, reals_unreals = self.get_overlaps(overlap)
        else:
            print('ATTN')
            reals, unreals, reals_unreals = self.get_overlaps(overlap)
        
        to_del = []
        for key in reals:
            print(key, reals[key])
            if len(reals[key]) == 0:
                to_del.append(key)
        for key in to_del:
            del reals[key]
            del unreals[key]
            del reals_unreals[key]
        
        categories = list(reals.keys())
        
        set1_counts = [statistics.mean([val for item in reals[category] for val in item]) for category in categories]
        set2_counts = [statistics.mean([val for item in unreals[category] for val in item]) for category in categories]
        set3_counts = [statistics.mean([val for item in reals_unreals[category] for val in item]) for category in categories]
        
        set1_stderr = [np.std([val for item in reals[category] for val in item]) for category in categories]
        set2_stderr = [np.std([val for item in unreals[category] for val in item]) for category in categories]
        set3_stderr = [np.std([val for item in reals_unreals[category] for val in item]) for category in categories]
        
        # # Plotting
        # plt.figure(figsize=(12, 15))
        # plt.bar([new_names[l] for l in list(langFile['type'].unique())], acc_base, color=colors)
        # plt.errorbar([new_names[l] for l in list(langFile['type'].unique())], y=acc_base, yerr=acc_stderr, fmt='none', c='black', capsize=5)

        print(set1_counts, set2_counts, set3_counts)
        # Number of categories
        num_categories = len(categories)
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(10, 10))
        
        bar_width = 0.3
        index = np.arange(num_categories)
        
        bars1 = ax.bar(index - bar_width, set1_counts, bar_width, label='HxH', color=self.real_color)
        err1 = ax.errorbar(index - bar_width, set1_counts, yerr=set1_stderr, fmt='none', c='black', capsize=5)
        bars2 = ax.bar(index, set2_counts, bar_width, label='LxL', color=self.unreal_color)
        err2 = ax.errorbar(index, set2_counts, yerr=set2_stderr, fmt='none', c='black', capsize=5)
        bars3 = ax.bar(index + bar_width, set3_counts, bar_width, label='HxL', color=self.mixed_color)
        err3 = ax.errorbar(index + bar_width, set3_counts, yerr=set3_stderr, fmt='none', c='black', capsize=5)
        
        # Add labels, title, and legend
        plt.grid(True, which='major', axis='y', linestyle='--', linewidth=2, color='gray', alpha=1)
        # ax.set_xlabel('Language')
        ax.set_ylabel(f'Mean Overlap Percentage')
        # ax.set_title(f'{component} overlaps', fontsize=21)
        ax.set_xticks(index)
        ax.set_yticks([0,0.25,0.5, 0.75,1])
        ax.set_xticklabels([key.upper() for key in reals.keys()])
        ax.legend(ncol=3, handlelength=1, handleheight=1, columnspacing=1, frameon=False)
        plt.tight_layout()
        plt.savefig(f'{self.op_dir}/{model_name}-{self.token_type}-mean-{component}-overlaps.png')
        # Display the plot
        plt.show()

class Expt2_1_nonce_conv:
    def __init__(self, config):
        self.get_files = config.get_files
        self.model_dirs = config.model_dirs_exp_2
        self.model_name = config.model_name
        self.token_type = config.token_type
        self.grammar_names = config.grammar_names
        self.attn_pickle_dirs = config.model_dirs_exp_2['attn']
        self.mlp_pickle_dirs = config.model_dirs_exp_2['mlp']
        self.op_dir = config.op_dirs['expt2.1']
        self.real_color = config.constants.real_color
        self.unreal_color = config.constants.unreal_color
        self.mixed_color = config.constants.mixed_color
        
    def create_plot(self):
        print('Confusion Matrix Plots')
        attn, attn_values = self.conf_matrix_plot(self.model_name, 'attn', 0.01, self.attn_pickle_dirs)
        mlp, mlp_values = self.conf_matrix_plot(self.model_name, 'mlp', 0.01, self.mlp_pickle_dirs)
        
        print('Mean plots')
        self.draw_overlap_bars_plot("attn", attn, self.model_name)
        self.draw_overlap_bars_plot("mlp", mlp, self.model_name)
        
        # print(f"Language x Structure {self.model_name}")
        # self.get_overlap_values_langxstructure(mlp_values[0], mlp, 'MLP')
        # self.get_overlap_values_langxstructure(attn_values[0], attn, 'ATTN')

        # print(f"Structure {self.model_name}")
        # self.get_overlap_values_structure(mlp_values[0], mlp, 'MLP')
        # self.get_overlap_values_structure(attn_values[0], attn, 'ATTN')
        
    def conf_matrix_plot(self, model, compPath, topK, dirs):
        def conf_matrix_individual(model, topK, pkl_dirs):
            columns = self.grammar_names.keys()
            reals = [col for col in sorted(columns) if not '_S-' in col]
            nonce = [col for col in sorted(columns) if '_S-' in col]
            components = pd.DataFrame(columns=sorted(reals+nonce))
            component_v = pd.DataFrame(columns=sorted(reals+nonce))
            
            paths = [f'{pkl_dirs[0]}/{col}.pkl' for col in reals] + [f'{pkl_dirs[1]}/{col}.pkl' for col in nonce]
            for _path in paths:
                # try:
                with open(_path, 'rb') as f:
                    col = _path.split('/')[-1].split('.pkl')[0]
                    topk_cache = pickle.load(f)
                    topk_cache = topk_cache.cpu()
                    flattened_effects_cache = topk_cache.view(-1)
                    top_neurons = flattened_effects_cache.topk(k=int(topK * flattened_effects_cache.shape[-1]))
                    two_d_indices = torch.cat((((top_neurons[1] // topk_cache.shape[1]).unsqueeze(1)), ((top_neurons[1] % topk_cache.shape[1]).unsqueeze(1))), dim=1)            
                    df = pd.DataFrame(two_d_indices, columns=['layer', 'neuron'])
                    two_d_indices_1 = torch.cat((((top_neurons[1] // topk_cache.shape[1]).unsqueeze(1)), ((top_neurons[1] % topk_cache.shape[1]).unsqueeze(1)), top_neurons[0].unsqueeze(1)), dim=1)            
                    df_1 = pd.DataFrame(two_d_indices_1, columns=['layer', 'neuron', 'value'])

                    components[col] = list(df.itertuples(index=False))
                    component_v[col] = list(df_1.itertuples(index=False))

            print('Gathered pickles')
            print(components.columns)
            # print(components['en-r-1'])
            overlap = pd.DataFrame(columns=reals, index=nonce)
            for col in tqdm(reals):
                for idx in nonce:
                    overlap.loc[(overlap.index == idx), col] = \
                        len(set(components[col]).intersection(set(components[idx])))
                overlap[col] = overlap[col].astype(float)
            ov_max = overlap.max().max()
            ov_min = overlap.min().min()
            def normalize(x):
                return (x - ov_min) / (ov_max - ov_min)
            overlap = overlap.applymap(normalize)

            # print('NUM NEURONS ', len(real_vals.columns))
            return overlap, component_v, reals, nonce
        
        overall_comp_path = []
        overall_comp_values = []
        reals  = 0
        
        overlap, component_v, reals, nonce = conf_matrix_individual(model, topK, dirs)
        overall_comp_path.append(overlap)
        overall_comp_values.append(component_v)
        
        average_df = pd.DataFrame(columns=reals, index=nonce)
        
        for row in nonce:
            for col in reals:
                average_df.loc[(average_df.index == row), col] = float(np.mean([overall_comp_path[i].loc[(overall_comp_path[i].index == row), col] for i in range(len(overall_comp_path))]))
        
        average_df = average_df.apply(pd.to_numeric)
        # print(average_df)
        print(reals)
        plt.figure(figsize=(10, 8))
        plt.imshow(overlap, cmap='binary', aspect='auto', interpolation='nearest')
        plt.xticks(range(len(overlap.columns)), labels=[self.grammar_names[col] for col in overlap.columns])
        plt.yticks(range(len(sorted(overlap.index))), labels=[self.grammar_names[col] for col in overlap.index])
        plt.xticks(fontsize=12, rotation=-45, ha='left')  # Set x-axis tick label font size
        plt.yticks(fontsize=12)  # Set y-axis tick label font size

        cbar = plt.colorbar()
        cbar.set_label('Overlap %', fontsize=15)
        cbar.ax.tick_params(labelsize=15) 
        
        plt.title(f'{model.title()} {compPath.title()}', fontsize=15)
        plt.tight_layout()

        # -------------------------------- Draw lines -------------------------------------------------
        plt.axvline(x=2.5, color='black')
        # plt.axvline(x=ux_tick[0], color=unreal_color)
        plt.axhline(y=2.5, color='black')
        # plt.axhline(y=ux_tick[0], color=unreal_color)
        plt.savefig(f'{self.op_dir}/{self.token_type}-{model}-{compPath}-overlap-{topK}.png')
        plt.show()
        return overall_comp_path, overall_comp_values
    
    def get_values_for_substr(self, substring, df):
        filtered_df = df[df.index.str.contains(substring)]
        layer_neuron_dict = {}
        for _, row in filtered_df.iterrows():
            items = [ r[1].split(',') for r in row.items() if type(r[1]) == str]
            for (layer, neuron, value) in items:
                layer = int(float(layer))
                neuron = int(float(neuron))
                value = float(value)
                if (layer, neuron) not in layer_neuron_dict:
                    layer_neuron_dict[f"{layer}-{neuron}"] = []
                layer_neuron_dict[f"{layer}-{neuron}"].append(value)
        # print(substring, len(layer_neuron_dict.keys()))
        return layer_neuron_dict

    def intersections(self, dict1, dict2):
        intersecting_keys = set(dict1.keys()).intersection(dict2.keys())
        # print(len(dict1.keys()), len(dict2.keys()), len(intersecting_keys))
        intersection = {}

        max_len = 0
        for key in intersecting_keys:
            list1 = dict1[key]
            list2 = dict2[key]
            combined_list = list1 + list2
            intersection[key] = combined_list
            max_len = max(len(intersection[key]), max_len)
        return intersection

    def get_overlap_values_langxstructure(self, comp_df, overlap, comp):
        reals, unreals, reals_unreals = self.get_overlaps(overlap)
        real_real_values = {key: [] for key in reals.keys() if len(reals[key]) > 0}
        unreal_unreal_values = {key: [] for key in reals.keys() if len(reals[key]) > 0}
        real_unreal_values = {key: [] for key in reals.keys() if len(reals[key]) > 0}

        for lang in real_real_values.keys():
            real_real_values[lang] = self.get_values_for_substr(f'en-', comp_df)
            unreal_unreal_values[lang] = self.get_values_for_substr(f'en_S-', comp_df)
            real_unreal_values[lang] = self.intersections(real_real_values[lang], unreal_unreal_values[lang])

        real_real_total = {key: [] for key in reals.keys() if len(reals[key]) > 0}
        unreal_unreal_total = {key: [] for key in reals.keys() if len(reals[key]) > 0}
        real_unreal_total_x_real = {key: [] for key in reals.keys() if len(reals[key]) > 0}
        real_unreal_total_x_unreal = {key: [] for key in reals.keys() if len(reals[key]) > 0}

        for lang in real_real_total:
            for key in real_real_values[lang].keys():
                for item in real_real_values[lang][key]:
                    real_real_total[lang].append(item)

            for key in unreal_unreal_values[lang].keys():
                for item in unreal_unreal_values[lang][key]:
                    unreal_unreal_total[lang].append(item)

            for key in real_unreal_values[lang].keys():
                list_len = len(real_unreal_values[lang][key])
                for item in real_unreal_values[lang][key][:list_len//2]:
                    real_unreal_total_x_real[lang].append(item)

            for key in real_unreal_values[lang].keys():
                list_len = len(real_unreal_values[lang][key])
                for item in real_unreal_values[lang][key][list_len//2:]:
                    real_unreal_total_x_unreal[lang].append(item)

        print(f"{'Component':<15} | {'LANG':<10} | {'CxN/CxC':<25} | {'CxN/NxN':<30}")
        print("-" * 85)
        for lang in real_real_values:
            real_unreal_real_ratio = sum(real_unreal_total_x_real[lang]) / sum(real_real_total[lang])
            real_unreal_unreal_ratio = sum(real_unreal_total_x_unreal[lang]) / sum(unreal_unreal_total[lang])

            print(f"{comp:<15} | {lang:<10} | {real_unreal_real_ratio: <25.2f} | {real_unreal_unreal_ratio: <30.2f}")
            
    def get_overlap_values_structure(self, comp_df, overlap, comp):

        real_values = []
        unreal_values = []


        real_values = self.get_values_for_substr(f'en-', comp_df)
        unreal_values = self.get_values_for_substr(f'en_S-', comp_df)
        real_unreal_values = self.intersections(real_values, unreal_values)

        # print(real_values)
        real_total = []
        unreal_total = []
        real_unreal_total_x_real = []
        real_unreal_total_x_unreal = []

        for key in real_values.keys():
            for item in real_values[key]:
                real_total.append(item)

        for key in unreal_values.keys():
            for item in unreal_values[key]:
                unreal_total.append(item)
        
        ctr = 0
        for key in real_unreal_values.keys():
            list_len = len(real_unreal_values[key])
            for item in real_unreal_values[key][:list_len//2]:
                real_unreal_total_x_real.append(item)
        for key in real_unreal_values.keys():
            list_len = len(real_unreal_values[key])
            for item in real_unreal_values[key][list_len//2:]:
                real_unreal_total_x_unreal.append(item)
                
        # Print table header
        print(f"{'Component':<15} | {'CxN/CxC':<25} | {'CxN/NxN':<30}")
        print("-" * 75)

        # Calculate and print ratios
        real_unreal_real_ratio = sum(real_unreal_total_x_real) / sum(real_total)
        real_unreal_unreal_ratio = sum(real_unreal_total_x_unreal) / sum(unreal_total)

        # Print formatted results
        print(f"{comp:<15} | {real_unreal_real_ratio:<25.2f} | {real_unreal_unreal_ratio:<25.2f}")

    def get_overlaps(self, conf_matrices):
        reals = { 'en': [], 'ita': [], 'jap': [] }
        unreals = { 'en': [], 'ita': [], 'jap': [] }
        reals_unreals = { 'en': [], 'ita': [], 'jap': [] }
        cols = conf_matrices[0].index
        for ind in cols:
            for col in conf_matrices[0].columns:
                r_en = []
                u_en = []
                ru_en = []
                
                for conf_matrix in conf_matrices:
                    if (not '-u-' in col) and (not '-u-' in ind):
                        r_en.append(conf_matrix[(conf_matrix.index == ind)][col].item())
                    elif ('-u-' in col) and ('-u-' in ind):
                        u_en.append(conf_matrix[(conf_matrix.index == ind)][col].item())
                    elif ('-u-' in col) and ('-r-' in ind):
                        ru_en.append(conf_matrix[(conf_matrix.index == ind)][col].item())
                if (not '-u-' in col) and (not '-u-' in ind):
                    reals['en'].append(r_en)
                elif ('-u-' in col) and ('-u-' in ind):    
                    unreals['en'].append(u_en)
                elif  ('-u-' in col) and ('-r-' in ind):
                    if ('en' in col[:2]) and ('en' in ind[:2]):
                        reals_unreals['en'].append(ru_en)
                    
        t_reals = [val for key in reals.keys() for val_list in reals[key] for val in val_list]
        t_unreals = [val for key in reals.keys() for val_list in unreals[key]  for val in val_list]
        t_reals_unreals = [val for key in reals.keys() for val_list in reals_unreals[key]  for val in val_list]
        print('C-C vs N-N', stats.mannwhitneyu(t_reals, t_unreals))
        print('C-C vs C-N', stats.mannwhitneyu(t_reals, t_reals_unreals))
        print('N-N vs C-N', stats.mannwhitneyu(t_unreals, t_reals_unreals))
        return reals, unreals, reals_unreals

    def draw_overlap_bars_plot(self, component, overlap, model_name):
        # Extract the categories and the values
        if component == 'MLP':
            print('MLP')
            reals, unreals, reals_unreals = self.get_overlaps(overlap)
        else:
            print('ATTN')
            reals, unreals, reals_unreals = self.get_overlaps(overlap)
        
        to_del = []
        for key in reals:
            print(key, reals[key])
            if len(reals[key]) == 0:
                to_del.append(key)
        for key in to_del:
            del reals[key]
            del unreals[key]
            del reals_unreals[key]
        
        categories = list(reals.keys())
        
        set1_counts = [statistics.mean([val for item in reals[category] for val in item]) for category in categories]
        set2_counts = [statistics.mean([val for item in unreals[category] for val in item]) for category in categories]
        set3_counts = [statistics.mean([val for item in reals_unreals[category] for val in item]) for category in categories]
        
        # set1_stderr = [np.std([val for item in reals[category] for val in item]) for category in categories]
        # set2_stderr = [np.std([val for item in unreals[category] for val in item]) for category in categories]
        # set3_stderr = [np.std([val for item in reals_unreals[category] for val in item]) for category in categories]
        
        # # Plotting
        # plt.figure(figsize=(12, 15))
        # plt.bar([new_names[l] for l in list(langFile['type'].unique())], acc_base, color=colors)
        # plt.errorbar([new_names[l] for l in list(langFile['type'].unique())], y=acc_base, yerr=acc_stderr, fmt='none', c='black', capsize=5)

        print(set1_counts, set2_counts, set3_counts)
        # Number of categories
        num_categories = len(categories)
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(10, 10))
        
        bar_width = 0.3
        index = np.arange(num_categories)
        
        ax.bar(index - bar_width, set1_counts, bar_width, label='CxC', color=self.real_color)
        # ax.errorbar(index - bar_width, set1_counts, yerr=set1_stderr, fmt='none', c='black', capsize=5)
        ax.bar(index, set2_counts, bar_width, label='NxN', color=self.unreal_color)
        # ax.errorbar(index, set2_counts, yerr=set2_stderr, fmt='none', c='black', capsize=5)
        ax.bar(index + bar_width, set3_counts, bar_width, label='CxN', color=self.mixed_color)
        # ax.errorbar(index + bar_width, set3_counts, yerr=set3_stderr, fmt='none', c='black', capsize=5)
        
        # Add labels, title, and legend
        plt.grid(True, which='major', axis='y', linestyle='--', linewidth=2, color='gray', alpha=1)
        # ax.set_xlabel('Language')
        ax.set_ylabel(f'Mean Overlap Percentage')
        # ax.set_title(f'{component} overlaps', fontsize=21)
        ax.set_xticks(index)
        ax.set_yticks([0,0.25,0.5, 0.75,1])
        ax.set_xticklabels([key.upper() for key in reals.keys()])
        ax.legend(ncol=3, handlelength=1, handleheight=1, columnspacing=1, frameon=False)
        plt.tight_layout()
        plt.savefig(f'{self.op_dir}/{model_name}-{self.token_type}-mean-{component}-overlaps.png')
        # Display the plot
        plt.show()


def main():
    MODEL_NAMES = ['gpt-2-xl', 'llama-2-7b', 'llama-3.1-8b', 'mistral-v0.3', 'qwen-2-0.5b', 'llama-3.1-70b', 'qwen-2-1.5b']
    TOKEN_TYPES = ['nonce', 'conventional', 'nonce_conv']
    
    for model_name in MODEL_NAMES:
        for token_type in TOKEN_TYPES[2:]:
            if token_type == 'nonce_conv':
                print(f"########## Creating experiment 2.1 plots for {model_name} and {token_type} tokens dataset")
                config = Config(model_name, token_type)
                expt_2_1 = Expt2_1_nonce_conv(config)
                expt_2_1.create_plot()
            else:
            # try:
                print(f"########## Creating experiment 1 plots for {model_name} and {token_type} tokens dataset")
                config = Config(model_name, token_type)
                expt_1 = Expt1(config)
                expt_1.create_plot()
                
                print(f"########## Creating experiment 2 plots for {model_name} and {token_type} tokens dataset")
                expt_2 = Expt2(config)
                expt_2.create_plot()
            # except Exception as e:
            #     print(f"Failed to create plots for {model_name} and {token_type} tokens dataset")
            #     print(e)
            #     continue
            
if __name__ == "__main__":
    main()
    