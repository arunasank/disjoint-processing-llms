import re

from numpy.random import f
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
from scipy.stats import ttest_ind, mannwhitneyu, shapiro
from scipy.stats import chi2_contingency
from scipy.stats import mannwhitneyu
from tabulate import tabulate
from scipy.stats import combine_pvalues
from scipy.stats import sem
import glob
import os
from statistics import mean
from matplotlib.ticker import FuncFormatter

# Custom formatter function
def custom_formatter(x, _):
    return f'{x:.2f}'.lstrip('0').rstrip('0').rstrip('.')

# Apply custom formatter globally
formatter = FuncFormatter(custom_formatter)

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
        
        self.model_labels = {
            'llama-2-7b': 'Llama-2 (7B)',
            'llama-3.1-8b': 'Llama-3.1 (8B)',
            'llama-3.1-70b': 'Llama-3.1 (70B)',
            'qwen-2-0.5b': 'Qwen-2 (0.5B)',
            'qwen-2-1.5b': 'Qwen-2 (1.5B)',
            'mistral-v0.3': 'Mistral-v0.3 (7B)'
        }

        self.nonce_names = {'en_S-r-1': 'ZZ Declarative (H)',
        'en_S-r-2-subordinate': 'ZZ Subordinate (H)',
        'en_S-r-3-passive': 'ZZ Passive (H)',
        'en_S-u-1-negation': 'ZZ Negation (L)',
        'en_S-u-2-inversion':'ZZ Inversion (L)',
        'en_S-u-3-wh':'ZZ Wh-word (L)',
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

        self.nonce_conv_names = {'en_S-r-1': 'EN Declarative (H) ZZ',
        'en_S-r-2-subordinate': 'EN Subordinate (H) ZZ',
        'en_S-r-3-passive': 'EN Passive (H) ZZ',
        'en_S-u-1-negation': 'EN Negation (L) ZZ',
        'en_S-u-2-inversion':'EN Inversion (L) ZZ',
        'en_S-u-3-wh':'EN Wh- word (L) ZZ',
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
    def __init__(self, model_names, token_type, expt_num):
        self.constants = Constants()
        self.model_names = model_names
        # print(self.model_names)
        self.token_type = token_type
        self.grammar_names = []
        if token_type == 'nonce':
            model_dirs = ['nonce-10']
            self.grammar_names = self.constants.nonce_names
        elif token_type == 'conventional':
            model_dirs = ['conventional-10']
            self.grammar_names = self.constants.conventional_names
        elif token_type == 'nonce_conv' and expt_num == 2:
            model_dirs = ['conventional-10', 'nonce-10']
            self.grammar_names = self.constants.nonce_conv_names
        elif token_type == 'nonce_conv' and expt_num == 3:
            model_dirs = ['conventional-10']
            self.grammar_names = self.constants.nonce_conv_names
        else:
            assert False, f"Enter the appropriate token type {token_type}"
        self.model_dirs_exp_1 = []
        self.model_dirs_exp_2 = { "attn": [], "mlp": [] }
        self.model_dirs_exp_3 = { "real": [], "unreal": [], "random": [] }
        for model_name in self.model_names:
            for m_dir in model_dirs:
                self.model_dirs_exp_1.append(f"/mnt/align4_drive/arunas/broca/{model_name}/experiments/{m_dir}/")
                self.model_dirs_exp_2["attn"].append(f"/mnt/align4_drive/arunas/broca/{model_name}/atp/patches/attn/{m_dir}")
                self.model_dirs_exp_2["mlp"].append(f"/mnt/align4_drive/arunas/broca/{model_name}/atp/patches/mlp/{m_dir}")
                if token_type in ["nonce", "conventional"]:
                    self.model_dirs_exp_3["real"].append(f"/mnt/align4_drive/arunas/broca/{model_name}/experiments/{m_dir}-real-grammar-specific-all/")
                    self.model_dirs_exp_3["unreal"].append(f"/mnt/align4_drive/arunas/broca/{model_name}/experiments/{m_dir}-unreal-grammar-specific-all/")
                    self.model_dirs_exp_3["random"].append(f"/mnt/align4_drive/arunas/broca/{model_name}/experiments/{m_dir}-random-grammar-specific-all/")
                elif token_type == "nonce_conv" and expt_num == 3:
                    self.model_dirs_exp_1.append(f"/mnt/align4_drive/arunas/broca/{model_name}/experiments/nonce-10/")
                    self.model_dirs_exp_3["real"].append(f"/mnt/align4_drive/arunas/broca/{model_name}/experiments/{m_dir}-ablate-conv-nonce-act-real-grammar-specific-all/")
                    self.model_dirs_exp_3["unreal"].append(f"/mnt/align4_drive/arunas/broca/{model_name}/experiments/{m_dir}-ablate-conv-nonce-act-unreal-grammar-specific-all/")
                    self.model_dirs_exp_3["random"].append(f"/mnt/align4_drive/arunas/broca/{model_name}/experiments/{m_dir}-ablate-conv-nonce-act-random-grammar-specific-all/")
        self.op_dirs = self.make_op_dirs()
        # print(self.model_dirs_exp_1)
        
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
            assert len(dir_list) != 0, f"No files found in {m_dir}"
            for d in dir_list:
                d = pd.read_csv(d).head(1)
                d['grammar'] = d['lang']
                d['real'] = d['grammar'].apply(lambda x: 0 if '-u-' in x else 1)
                if self.token_type == 'conventional':
                    d['language'] = d['grammar'].apply(lambda x: x.split('-')[0].upper())
                elif self.token_type == 'nonce':
                    d['language'] = d['grammar'].apply(lambda x: x.split('_S')[0].upper())
                elif self.token_type == 'nonce_conv':
                    d['language'] = d['grammar'].apply(lambda x: x.split('_S')[0].upper())
                acc = pd.concat([acc, pd.DataFrame.from_dict({'acc': d['acc'], 'language': d['language'], 'grammar': d['grammar'], 'real': d['real']})])
            acc = acc[acc['grammar'].isin(self.grammar_names.keys())]
            acc = acc.sort_values(by='grammar', key=lambda x: x.str.lower())
            acc.loc[(acc['language'] == 'JAP'),'language'] = 'JP'
            acc.loc[(acc['language'] == 'ITA'),'language'] = 'IT'
            assert len(acc) > 0, f"Empty dataframe for {m_dir}"
            
            return acc
     
class Expt1:
    def __init__(self, config):
        self.get_files = config.get_files
        self.model_dirs = config.model_dirs_exp_1
        self.model_names = config.model_names
        self.token_type = config.token_type
        self.grammar_names = config.grammar_names
        self.constants = config.constants
        self.op_dir = config.op_dirs['expt1']
        self.labels = config.constants.model_labels
    
    def create_plot(self):
        self.full_model_df = pd.DataFrame(columns=['model_name', 'acc', 'lang'])
        for model_name in self.model_names:
            self.model_name = model_name
            self.model_df = self.create_df(model_name)
            self.full_model_df = pd.concat([self.full_model_df, self.model_df])
        
        self.create_plot_lang()
        self.create_plot_grammar()

        self.create_plot_lang_avg()
        self.create_plot_grammar_avg()
        self.create_all_models_same_plot()
        
        self.tables()
        
    def tables(self):

        def get_stat_sig(df):
            results = []
            for lang in df['language'].unique():
                hierarchical = df[(df['real'] == 1) & (df['language'] == lang)]['acc']
                linear = df[(df['real'] == 0) & (df['language'] == lang)]['acc']
                # Check normality
                _, p_A = shapiro(hierarchical)
                _, p_B = shapiro(linear)
                # Select appropriate test
                if p_A > 0.05 and p_B > 0.05:
                    print('t test')
                    stat, p_value = ttest_ind(hierarchical, linear)
                else:
                    print('mwu test')
                    stat, p_value = mannwhitneyu(hierarchical, linear)
                results.append([lang, f"{stat:.4f}", f"{p_value:.4f}"])

            latex_table = tabulate(results, headers=['Language', 'Test-Statistic', 'P-value'], tablefmt='latex')
            print('##### Experiment 1: Statistical Significance')
            print(latex_table)
        
        def generate_latex_table(df):
            latex_rows = []
            for model in df['model_name'].unique():
                model_df = df[df['model_name'] == model].sort_values(by=['language', 'grammar'])
                for lang in model_df['language'].unique():
                    lang_df = model_df[model_df['language'] == lang]
                    first_row = True
                    for _, row in lang_df.iterrows():
                        model_cell = f"\\multirow{{{len(lang_df)}}}{{*}}{{{self.labels[model]}}}" if first_row else ""
                        lang_cell = f"\\multirow{{{len(lang_df)}}}{{*}}{{{lang}}}" if first_row else ""
                        latex_rows.append([model_cell, lang_cell, self.grammar_names[row['grammar']], f"{row['acc']:.2f}"])
                        first_row = False
            headers = ['Model Name', 'Language', 'Grammar', 'Accuracy']
            latex_table = tabulate(latex_rows, headers=headers, tablefmt='latex_raw', stralign='center')
            print(latex_table)

        get_stat_sig(self.full_model_df)
        generate_latex_table(self.full_model_df)
     
    def create_plot_grammar_avg(self):
        plt.rcParams.update({'font.size': 21})
        np.set_printoptions(formatter={'float': lambda x: f'{x:.2f}'.rstrip('0').rstrip('.')})
        grammar = self.grammar_names.keys()
        avg_df = self.full_model_df[self.full_model_df['grammar'].isin(grammar)].groupby('grammar')['acc'].mean()
        colors = []
        for grammar in avg_df.index:
            if '-r-' in grammar:
                colors.append(self.constants.real_color)
            else:
                colors.append(self.constants.unreal_color)

        avg_df = avg_df.sort_index(ascending=True)
        plt.figure(figsize=(12, 15))
        plt.bar(avg_df.index, avg_df.values, color=colors, edgecolor='black')
        plt.ylabel('Average Accuracy')
        plt.ylim(0, 1)

        # Creating legend handles
        real_patch = plt.Line2D([0], [0], color=self.constants.real_color, lw=4, label='H')
        unreal_patch = plt.Line2D([0], [0], color=self.constants.unreal_color, lw=4, label='L')

        plt.xticks(range(len(self.grammar_names.keys())), labels=self.grammar_names.values(), rotation=90)
        plt.legend(handles=[real_patch, unreal_patch], ncol=2)
        plt.tight_layout()
        plt.savefig(f'{self.op_dir}/{self.op_dir.split('/')[-1]}-{self.token_type}-average-grammars.png', dpi=300)
        plt.show()

    def create_plot_lang_avg(self):
        global formatter
        plt.rcParams.update({'font.size': 15})
        np.set_printoptions(formatter={'float': lambda x: f'{x:.2f}'.rstrip('0').rstrip('.')})
        lang = self.full_model_df['language'].unique()
        r_data = []
        u_data = []

        for l in lang:
            r_data.append(self.full_model_df[(self.full_model_df['real'] == 1) & (self.full_model_df['language'] == l)]['acc'].tolist())
            u_data.append(self.full_model_df[(self.full_model_df['real'] == 0) & (self.full_model_df['language'] == l)]['acc'].tolist())

        r_means = [statistics.mean(rd) for rd in r_data]
        u_means = [statistics.mean(ud) for ud in u_data]
        r_var = [sem(rd) for rd in r_data]
        u_var = [sem(ud) for ud in u_data]
        
        fig, ax = plt.subplots(figsize=(10, 15))
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        bar_width = 0.2
        index = np.arange(len(lang))

        ax.bar(index, r_means, bar_width, label='H', color=[self.constants.real_color], edgecolor='black')
        ax.bar(index + bar_width, u_means, bar_width, label='L', color=[self.constants.unreal_color], edgecolor='black')

        plt.grid(True, which='major', axis='y', linestyle='--', linewidth=1.5, color='gray', alpha=1)
        plt.errorbar(index, r_means, yerr=r_var, fmt='none', c='black', capsize=5)
        plt.errorbar(index + bar_width, u_means, yerr=u_var, fmt='none', c='black', capsize=5)
        ax.set_ylabel('Mean Accuracy')
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(lang)
        ax.legend(frameon=False, ncol=2)
        plt.tight_layout()
        fig.savefig(f'{self.op_dir}/{self.op_dir.split('/')[-1]}-{self.token_type}-average-lang.png', dpi=300)
        plt.show()
    
    def create_plot_grammar(self):
        plt.rcParams.update({'font.size': 21})
        np.set_printoptions(formatter={'float': lambda x: f'{x:.2f}'.rstrip('0').rstrip('.')})
        model_names = self.model_names
        fig, axes = plt.subplots(1, len(model_names), figsize=(8 * len(model_names) + 1, 20), sharex=True, sharey=True)
        # Extract the categories and the values
        for model, ax in zip(model_names, axes):
            grammar = self.grammar_names.keys()
            colors = []
            for idx, row in self.full_model_df[(self.full_model_df['grammar'].isin(grammar)) & (self.full_model_df['model_name'] == model)].iterrows():
                colors.append(self.constants.real_color if row['real'] == 1 else self.constants.unreal_color)
                
            # plt.figure(figsize=(12, 15))
            accuracies = self.full_model_df[(self.full_model_df['grammar'].isin(grammar)) & (self.full_model_df['model_name'] == model)]['acc']
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)
            ax.bar(sorted(self.grammar_names.values()), accuracies, color=colors, edgecolor='black')
            ax.set_ylabel('Accuracy')
            ax.set_ylim(0,1)
            ax.set_title(self.labels[model])
            ax.set_xticks(range(len(self.grammar_names.keys())))
            ax.set_xticklabels(self.grammar_names.values(), rotation=90)
        # Creating legend handles
        real_patch = plt.Line2D([0], [0], color=self.constants.real_color, lw=4, label='H')
        unreal_patch = plt.Line2D([0], [0], color=self.constants.unreal_color, lw=4, label='L')
        axes[-1].legend(handles=[real_patch, unreal_patch], loc='upper left', ncol=1, bbox_to_anchor=(1, 1))
        plt.tight_layout()
        # plt.tight_layout()
        fig.savefig(f'{self.op_dir}/{self.op_dir.split('/')[-1]}-{self.token_type}-all-models-grammars.png', dpi=300)
        plt.show()

    def create_plot_lang(self):
        plt.rcParams.update({'font.size': 21})
        np.set_printoptions(formatter={'float': lambda x: f'{x:.2f}'.rstrip('0').rstrip('.')})
        model_names = self.model_names
        fig, axes = plt.subplots(1, len(model_names), figsize=(5 * len(model_names) + 1, 5), sharex=True, sharey=True)
        # Extract the categories and the values
        for model, ax in zip(model_names, axes):
            languages = sorted(self.full_model_df['language'].unique())
            r_data = []
            u_data = []
            for l in languages:
                r_data.append(self.full_model_df[(self.full_model_df['real'] == 1) & (self.full_model_df['language'] == l) & (self.full_model_df['model_name'] == model)]['acc'].tolist())
                u_data.append(self.full_model_df[(self.full_model_df['real'] == 0) & (self.full_model_df['language'] == l) & (self.full_model_df['model_name'] == model)]['acc'].tolist())

            r_means = [statistics.mean(rd) for rd in r_data]
            u_means = [statistics.mean(ud) for ud in u_data]
            r_var = [sem(rd) for rd in r_data]
            u_var = [sem(ud) for ud in u_data]
            bar_width = 0.2
            index = np.arange(len(languages))

            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)
            ax.bar(index, r_means, bar_width, label='H', color=[self.constants.real_color], edgecolor='black')
            ax.bar(index + bar_width, u_means, bar_width, label='L', color=[self.constants.unreal_color], edgecolor='black')
            ax.errorbar(index, r_means, yerr=r_var, fmt='none', c='black', capsize=5)
            ax.errorbar(index + bar_width, u_means, yerr=u_var, fmt='none', c='black', capsize=5)
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.set_xticks(index + bar_width / 2)
            ax.set_xticklabels(languages)
            ax.set_title(self.labels[model])
        # Creating legend handles
        real_patch = plt.Line2D([0], [0], color=self.constants.real_color, lw=4, label='H')
        unreal_patch = plt.Line2D([0], [0], color=self.constants.unreal_color, lw=4, label='L')
        axes[-1].legend(handles=[real_patch, unreal_patch], ncol=1, loc='upper left', bbox_to_anchor=(1, 1))
        axes[0].set_ylabel('Mean Accuracy')
        plt.tight_layout()
        fig.savefig(f'{self.op_dir}/{self.op_dir.split('/')[-1]}-{self.token_type}-all-models-lang.png', dpi=300)
    
    def create_df(self, model_name):
        model_df = []
        # print("### EXPT1: ", model_name, self.model_dirs)
        for m_dir in self.model_dirs:
            if not model_name in m_dir:
                continue
            model_df.append(self.get_files(m_dir))
        model_df = pd.concat(model_df)
        model_df['model_name'] = model_name
        model_df = model_df.reset_index()
        
        return model_df
        
    def create_all_models_same_plot(self):
        plt.rcParams.update({'font.size': 14})
        np.set_printoptions(formatter={'float': lambda x: f'{x:.2f}'.rstrip('0').rstrip('.')})
        markers = ['o', 's', '^', '+', 'p', 'h']   # Different marker styles '^', 'p', '*', 
        colors = ['skyblue', 'green', 'green', 'orange', 'orange', 'purple']
        # Get the unique languages
        if self.token_type == 'conventional':
            languages = self.full_model_df['language'].unique()
        else:
            languages = ['EN']

        # Create subplots: one subplot per language'
        if self.token_type == 'conventional':
            fig, axes = plt.subplots(1, len(languages), figsize=(4 * len(languages) + 1, 4), sharex=True, sharey=True)
        elif self.token_type == 'nonce':
            fig, axes = plt.subplots(1, len(languages), figsize=(4 * len(languages) + 1, 4), sharex=True, sharey=True)
        # If only one subplot (1 language), convert axes to a list for iteration
        if len(languages) == 1:
            axes = [axes]
            
        # Iterate through each language and plot
        for ax, lang in zip(axes, languages):
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)
            ax.set_xlim(0, 1.1)
            ax.set_ylim(0, 1.1)
            ax.plot([0, 1.1], [0, 1.1], linestyle='--', color='gray', linewidth=1)
            ax.set_xticks([0, 0.5, 0.75, 1])
            ax.set_yticks([0, 0.5, 0.75, 1])
            if self.token_type == 'conventional':
                ax.set_title(f'{lang}')
            else:
                ax.set_title(f'ZZ')
            model_average_x = []
            model_average_y = []
            for i in range(len(self.model_names)):
                x = list(self.full_model_df[
                    (self.full_model_df['language'] == lang) &
                    (self.full_model_df['model_name'] == self.model_names[i]) & 
                    # (self.full_model_df['grammar'].isin(self.grammar_names.keys())) &
                    (self.full_model_df['real'] == 0)
                ]['acc'])
                
                y = list(self.full_model_df[
                    (self.full_model_df['language'] == lang) &
                    (self.full_model_df['model_name'] == self.model_names[i]) & 
                    # (self.full_model_df['grammar'].isin(self.grammar_names.keys())) &
                    (self.full_model_df['real'] == 1)
                ]['acc'])
                
                model_average_x += x
                model_average_y += y
                scatter = ax.scatter(
                    mean(x), mean(y), 
                    marker=markers[i % len(markers)], 
                    color=colors[i % len(colors)], 
                    s=200, alpha=0.7, label=self.labels[self.model_names[i]]
                )
                # # Append handles and labels for legend
                # if len(handles) < len(self.model_names):  # Ensure only unique labels
                #     handles.append(scatter)
                #     labels.append(self.model_names[i])
            ax.scatter(mean(model_average_x), mean(model_average_y), 
                       marker='*', color='black', s=100, alpha=0.9,
                       label='Average across \nmodels')
            ax.grid(
                True, 
                which='major',    # Only major grid lines
                color='lightgray', 
                linestyle='-', 
                linewidth=0.5
            )

            
        if self.token_type == 'conventional':
            axes[2].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=14, frameon=False, ncol=1)
            axes[1].set_xlabel('Mean accuracy on Linear Grammars')
            plt.tight_layout(rect=[0.05, 0, 1, 1])
        elif self.token_type == 'nonce':
            axes[0].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10, frameon=False, ncol=1)
            axes[0].set_xlabel('Mean accuracy on \n Linear Grammars')

            # Adjust tight_layout
            plt.tight_layout(rect=[0.1, 0, 0.9, 1])  # Expand the layout range
        axes[0].set_ylabel('Mean accuracy on \n Hierarchical Grammars')

        # Save the figure
        plt.savefig(f'{self.op_dir}/{self.op_dir.split('/')[-1]}-{self.token_type}-all-models-accuracies.png', dpi=300)
        plt.close()

class Expt2:
    def __init__(self, config):
        self.model_dirs = config.model_dirs_exp_2
        self.model_names = config.model_names
        self.token_type = config.token_type
        self.grammar_names = config.grammar_names
        self.attn_pickle_dirs = config.model_dirs_exp_2['attn']
        self.mlp_pickle_dirs = config.model_dirs_exp_2['mlp']
        self.op_dir = config.op_dirs['expt2']
        self.real_color = config.constants.real_color
        self.unreal_color = config.constants.unreal_color
        self.mixed_color = config.constants.mixed_color
        self.labels = config.constants.model_labels
        
    def create_plot(self):
        # self.draw_all_plots()
        self.draw_overlap_bars_plot()
        
    def conf_matrix_plot(self, compPath, topK):
        def conf_matrix_individual(topK, pkl_dir):
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
            # print(index)
            print('Gathered pickles')
            component = component[list(component.columns)[:index]]
            fig_column_order = [col for col in component.index if not '-u-' in col] + [col for col in component.index if '-u-' in col]
            component_overlap = pd.DataFrame(columns=fig_column_order, index=fig_column_order)

                
            for col in fig_column_order:
                for col1 in fig_column_order:
                    component_overlap.loc[(component_overlap.index == col), col1] = \
                        len(set(component.loc[col]).intersection(set(component.loc[col1])))/len(component.columns)
                component_overlap[col] = component_overlap[col].astype(float)

            # print('NUM NEURONS ', len(component.columns))
            return component_overlap, component_values, fig_column_order
        
        fig_column_order  = 0
        model_names = self.model_names
        all_comp_overlap = []
        all_comp_values = []
        fig, axes = plt.subplots(len(model_names)//2, 2, figsize=(10 * 2 + 1, 4 * len(model_names)), sharex=True, sharey=True)
        axes = axes.ravel()
        for ax, model_name in zip(axes, model_names):
            overall_comp_overlap = []
            overall_comp_values = []
            if compPath == "attn":
                _dirs = [_dir for _dir in self.attn_pickle_dirs if model_name in _dir]
            elif compPath == "mlp":
                _dirs = [_dir for _dir in self.mlp_pickle_dirs if model_name in _dir]
            else:
                assert False, f"Enter the appropriate component path {compPath}"

            for _dir in _dirs:
                component_overlap, component_values, fig_column_order = conf_matrix_individual(topK, _dir)
                overall_comp_overlap.append(component_overlap)
                overall_comp_values.append(component_values)
                all_comp_overlap.append(component_overlap)
                all_comp_values.append(component_values)
            
            average_df = pd.DataFrame(columns=fig_column_order, index=fig_column_order)
            
            for row in fig_column_order:
                for col in fig_column_order:
                    average_df.loc[(average_df.index == row), col] = float(np.mean([overall_comp_overlap[i].loc[(overall_comp_overlap[i].index == row), col] for i in range(len(overall_comp_overlap))]))
            
            average_df = average_df.apply(pd.to_numeric)
            # ax.xaxis.set_major_formatter(formatter)
            # ax.yaxis.set_major_formatter(formatter)
            im = ax.imshow(average_df, cmap='binary', aspect='auto', interpolation='nearest')
            ax.set_yticks(range(len(sorted(average_df.index))), labels=[self.grammar_names[col] for col in fig_column_order], fontsize=18, rotation=45)
            
            ax.set_title(f'{self.labels[model_name]} {compPath.title()}', fontsize=15)
            # -------------------------------- Draw lines -------------------------------------------------
            if self.token_type == 'conventional':
                ax.axvline(x=8.5, color='black')
                ax.axhline(y=8.5, color='black')
            elif self.token_type == 'nonce':
                ax.axvline(x=2.5, color='black')
                ax.axhline(y=2.5, color='black')
        axes[-1].set_xticks(range(len(fig_column_order)), labels=[self.grammar_names[col] for col in fig_column_order], fontsize=18, rotation=-45, ha='left')
        axes[-2].set_xticks(range(len(fig_column_order)), labels=[self.grammar_names[col] for col in fig_column_order], fontsize=18, rotation=-45, ha='left')
        fig.colorbar(im, ax=axes[1], orientation='vertical', label='Overlap %')
        fig.colorbar(im, ax=axes[3], orientation='vertical', label='Overlap %')
        fig.colorbar(im, ax=axes[5], orientation='vertical', label='Overlap %')
        fig.tight_layout(rect=[0, 0, 1, 1])
        fig.savefig(f'{self.op_dir}/{self.op_dir.split('/')[-1]}-{self.token_type}-{compPath}-overlap-{topK}.png')
        return all_comp_overlap, all_comp_values
        
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

    def get_overlaps(self, conf_matrix, model_name):
        reals = { 'en': [], 'it': [], 'jp': [] }
        unreals = { 'en': [], 'it': [], 'jp': [] }
        reals_unreals = { 'en': [], 'it': [], 'jp': [] }
        cols = conf_matrix.index
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
                
                if (not '-u-' in col) and (not '-u-' in ind):
                    if ('en' in col[:2]) and ('en' in ind[:2]):
                        # print("conf matrix ", conf_matrix.index, conf_matrix.columns)
                        r_en.append(conf_matrix[(conf_matrix.index == ind)][col].item())
                    elif ('it' in col[:3]) and ('it' in ind[:3]):
                        r_ita.append(conf_matrix[(conf_matrix.index == ind)][col].item())
                    elif ('jap' in col[:3]) and ('jap' in ind[:3]):
                        r_jap.append(conf_matrix[(conf_matrix.index == ind)][col].item())
                elif ('-u-' in col) and ('-u-' in ind):
                    if ('en' in col[:2]) and ('en' in ind[:2]):
                        u_en.append(conf_matrix[(conf_matrix.index == ind)][col].item())
                    elif ('it' in col[:3]) and ('it' in ind[:3]):
                        u_ita.append(conf_matrix[(conf_matrix.index == ind)][col].item())
                    elif ('jap' in col[:3]) and ('jap' in ind[:3]):
                        u_jap.append(conf_matrix[(conf_matrix.index == ind)][col].item())
                elif ('-u-' in col) and ('-r-' in ind):
                    if ('en' in col[:2]) and ('en' in ind[:2]):
                        ru_en.append(conf_matrix[(conf_matrix.index == ind)][col].item())
                    elif ('it' in col[:3]) and ('it' in ind[:3]):
                        ru_ita.append(conf_matrix[(conf_matrix.index == ind)][col].item())
                    elif ('jap' in col[:3]) and ('jap' in ind[:3]):
                        ru_jap.append(conf_matrix[(conf_matrix.index == ind)][col].item())
                if (not '-u-' in col) and (not '-u-' in ind):
                    if ('en' in col[:2]) and ('en' in ind[:2]):
                        reals['en'].append(r_en)
                    elif ('it' in col[:3]) and ('it' in ind[:3]):
                        reals['it'].append(r_ita)
                    elif ('jap' in col[:3]) and ('jap' in ind[:3]):
                        reals['jp'].append(r_jap)
                elif ('-u-' in col) and ('-u-' in ind):    
                    if ('en' in col[:2]) and ('en' in ind[:2]):
                        unreals['en'].append(u_en)
                    elif ('it' in col[:3]) and ('it' in ind[:3]):
                        unreals['it'].append(u_ita)
                    elif ('jap' in col[:3]) and ('jap' in ind[:3]):
                        unreals['jp'].append(u_jap)
                elif  ('-u-' in col) and ('-r-' in ind):
                    if ('en' in col[:2]) and ('en' in ind[:2]):
                        reals_unreals['en'].append(ru_en)
                    elif ('it' in col[:3]) and ('it' in ind[:3]):
                        reals_unreals['it'].append(ru_ita)
                    elif ('jap' in col[:3]) and ('jap' in ind[:3]):
                        reals_unreals['jp'].append(ru_jap)
        return reals, unreals, reals_unreals

    def stat_sig(self, reals, unreals, reals_unreals, component, lang):
        t_reals = [val for key in reals.keys() for val_list in reals[key] for val in val_list]
        t_unreals = [val for key in reals.keys() for val_list in unreals[key]  for val in val_list]
        t_reals_unreals = [val for key in reals.keys() for val_list in reals_unreals[key]  for val in val_list]
        print('REALS ', len(t_reals), len(t_unreals), len(t_reals_unreals))
        print(f'############## {component} {lang} ################')
        print('H-H vs L-L', stats.mannwhitneyu(t_reals, t_unreals))
        print('H-H vs H-L', stats.mannwhitneyu(t_reals, t_reals_unreals))
        print('L-L vs H-L', stats.mannwhitneyu(t_unreals, t_reals_unreals))
        
    def draw_overlap_bars_plot(self):
        model_names = self.model_names
        img_prefix = 'across-models'
        m_avg_reals = {}
        m_avg_unreals = {}
        m_avg_reals_unreals = {}
        fig, axes = plt.subplots(1, len(model_names), figsize=(5 * len(model_names) + 1, 5), sharex=True, sharey=True)
        languages = []
        # Extract the categories and the values self.
        a_overlap, a_values = self.conf_matrix_plot("attn", 0.01)
        m_overlap, m_values = self.conf_matrix_plot("mlp", 0.01)
        ctr = 0
        for model, ax in zip(model_names, axes):
            a_reals, a_unreals, a_reals_unreals = self.get_overlaps(a_overlap[ctr], model)
            m_reals, m_unreals, m_reals_unreals = self.get_overlaps(m_overlap[ctr], model)            
            reals = {}
            unreals = {}
            reals_unreals = {}
            for lang in m_reals:
                # print('M REALS LANG ', m_reals[lang])
                # print('A REALS LANG', a_reals[lang])
                if len(m_reals[lang]) > 0:
                    if not lang in m_avg_reals:
                        m_avg_reals[lang] = []
                        m_avg_unreals[lang] = []
                        m_avg_reals_unreals[lang] = []
                    reals[lang] = m_reals[lang] + a_reals[lang]
                    unreals[lang] = m_unreals[lang] + a_unreals[lang]
                    reals_unreals[lang] = m_reals_unreals[lang] + a_reals_unreals[lang]
                    
                    m_avg_reals[lang] += reals[lang]
                    m_avg_unreals[lang] += unreals[lang]
                    m_avg_reals_unreals[lang] += reals_unreals[lang]
            
            languages = list(reals.keys())
            
            reals_bar = [mean([val for item in reals[lang] for val in item]) for lang in languages]
            unreals_bar = [mean([val for item in unreals[lang] for val in item]) for lang in languages]
            reals_unreals_bar = [mean([val for item in reals_unreals[lang] for val in item]) for lang in languages]
            
            reals_var = [sem([val for item in reals[lang] for val in item]) for lang in languages]
            unreals_var = [sem([val for item in unreals[lang] for val in item]) for lang in languages]
            reals_unreals_var = [sem([val for item in reals_unreals[lang] for val in item]) for lang in languages]
            
            num_langs = len(languages)
            
            bar_width = 0.2
            index = np.arange(num_langs)
            
            ax.bar(index - bar_width, reals_bar, bar_width, label='HxH', color=self.real_color, edgecolor='black')
            ax.errorbar(index - bar_width, reals_bar, yerr=reals_var, fmt='none', c='black', capsize=5)
            ax.bar(index, unreals_bar, bar_width, label='LxL', color=self.unreal_color, edgecolor='black')
            ax.errorbar(index, unreals_bar, yerr=unreals_var, fmt='none', c='black', capsize=5)
            ax.bar(index + bar_width, reals_unreals_bar, bar_width, label='HxL', color=self.mixed_color, edgecolor='black')
            ax.errorbar(index + bar_width, reals_unreals_bar, yerr=reals_unreals_var, fmt='none', c='black', capsize=5)
            
            # Add labels, title, and legend
            # plt.set_grid(True, which='major', axis='y', linestyle='--', linewidth=2, color='gray', alpha=1)
            # ax.set_xlabel('Language')
            # ax.set_title(f'{component} overlaps', fontsize=21)
            ax.set_title(self.labels[model])
            ax.set_xticks(index)
            if self.token_type == 'conventional':
                ax.set_xticklabels([key.upper() for key in reals.keys()])
            else:
                ax.set_xticklabels([f'ZZ' for key in reals.keys()])
            ctr += 1
        axes[0].set_yticks([0, 0.25, 0.5, 0.75,1, 1.1])
        axes[0].set_ylabel(f'Mean Overlap \n Percentage')
        axes[-1].legend(ncol=1, handlelength=1, handleheight=1, columnspacing=1, frameon=False, bbox_to_anchor=(1, 1), loc='upper left')
        fig.savefig(f'{self.op_dir}/{self.op_dir.split('/')[-1]}-{img_prefix}-{self.token_type}-mean-overlaps.png')
        
        for lang in m_avg_reals.keys():
            self.stat_sig(m_avg_reals, m_avg_unreals, m_avg_reals_unreals, 'attn+mlp', lang)
        m_reals_bar = [mean([val for item in m_avg_reals[lang] for val in item]) for lang in languages]
        m_unreals_bar = [mean([val for item in m_avg_unreals[lang] for val in item]) for lang in languages]
        m_reals_unreals_bar = [mean([val for item in m_avg_reals_unreals[lang] for val in item]) for lang in languages]
        
        m_reals_var = [sem([val for item in m_avg_reals[lang] for val in item]) for lang in languages]
        m_unreals_var = [sem([val for item in m_avg_unreals[lang] for val in item]) for lang in languages]
        m_reals_unreals_var = [sem([val for item in m_avg_reals_unreals[lang] for val in item]) for lang in languages]
        
        num_langs = len(languages)
            
        bar_width = 0.2
        index = np.arange(num_langs)
        if self.token_type == 'conventional':
            plt.figure(figsize=(7, 4))
            fontsize = 15    
        elif self.token_type == 'nonce':
            plt.figure(figsize=(7, 7))
            fontsize = 21
        with plt.rc_context({'font.size': fontsize}):
            plt.bar(index - bar_width, m_reals_bar, bar_width, label='HxH', color=self.real_color, edgecolor='black')
            plt.errorbar(index - bar_width, m_reals_bar, yerr=m_reals_var, fmt='none', c='black', capsize=5)
            plt.bar(index, m_unreals_bar, bar_width, label='LxL', color=self.unreal_color, edgecolor='black')
            plt.errorbar(index, m_unreals_bar, yerr=m_unreals_var, fmt='none', c='black', capsize=5)
            plt.bar(index + bar_width, m_reals_unreals_bar, bar_width, label='HxL', color=self.mixed_color, edgecolor='black')
            plt.errorbar(index + bar_width, m_reals_unreals_bar, yerr=m_reals_unreals_var, fmt='none', c='black', capsize=5)
            
            # plt.title("Average overlaps \n across models")
            if self.token_type == 'conventional':
                plt.xticks(index, labels=[key.upper() for key in languages])
            else:
                plt.xticks(index, labels=[f'ZZ' for key in languages])
            plt.yticks([0, 0.25, 0.5, 0.75,1])
            plt.ylabel('Mean Overlap \n Percentage')
            if self.token_type == 'conventional':
                plt.legend(bbox_to_anchor=(0.5, 1.15), loc='lower center', ncol=3, frameon=False)
            elif self.token_type == 'nonce':
                plt.legend()
            plt.tight_layout(rect=[0, 0, 0.9, 1])
            plt.savefig(f'{self.op_dir}/{self.op_dir.split('/')[-1]}-model-mean-{self.token_type}-overlaps.png')
        
class Expt2_1_nonce_conv:
    def __init__(self, config):
        self.model_dirs = config.model_dirs_exp_2
        self.model_names = config.model_names
        self.token_type = config.token_type
        self.grammar_names = config.grammar_names
        self.attn_pickle_dirs = config.model_dirs_exp_2['attn']
        self.mlp_pickle_dirs = config.model_dirs_exp_2['mlp']
        self.op_dir = config.op_dirs['expt2.1']
        self.real_color = config.constants.real_color
        self.unreal_color = config.constants.unreal_color
        self.mixed_color = config.constants.mixed_color
        self.labels = config.constants.model_labels
        
    def create_plot(self):
        # print('Confusion Matrix Plots')
        # attn, attn_values = self.conf_matrix_plot(self.model_name, 'attn', 0.01, self.attn_pickle_dirs)
        # mlp, mlp_values = self.conf_matrix_plot(self.model_name, 'mlp', 0.01, self.mlp_pickle_dirs)
        
        # print('Mean plots')
        # self.draw_overlap_bars_plot("attn", attn, self.model_name)
        # self.draw_overlap_bars_plot("mlp", mlp, self.model_name)
        
        # print(f"Language x Structure {self.model_name}")
        # self.get_overlap_values_langxstructure(mlp_values[0], mlp, 'MLP')
        # self.get_overlap_values_langxstructure(attn_values[0], attn, 'ATTN')

        # print(f"Structure {self.model_name}")
        # self.get_overlap_values_structure(mlp_values[0], mlp, 'MLP')
        # self.get_overlap_values_structure(attn_values[0], attn, 'ATTN')
        self.draw_overlap_bars_plot()
        
    def conf_matrix_plot(self, compPath, topK):
        def conf_matrix_individual(topK, pkl_dirs):
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
            # print(components.columns)
            # print(components['en-r-1'])
            overlap = pd.DataFrame(columns=reals, index=nonce)
            for col in reals:
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
        
        model_names = self.model_names
        all_comp_overlap = []
        all_comp_values = []
        
        reals  = 0
        nonce = 0
        fig, axes = plt.subplots(len(model_names)//2, 2, figsize=(10 * 2 + 1, 4 * len(model_names)), sharex=True, sharey=True)
        axes = axes.ravel()
        for model, ax in zip(model_names, axes):
            overall_comp = []
            overall_comp_values = []
            if compPath == "attn":
                _dirs = [
                    (_dir1, _dir2) 
                    for _dir1, _dir2 in zip(self.attn_pickle_dirs[::2], self.attn_pickle_dirs[1::2]) 
                    if model in _dir1 and model in _dir2
                ]

            elif compPath == "mlp":
                _dirs = [
                    (_dir1, _dir2) 
                    for _dir1, _dir2 in zip(self.attn_pickle_dirs[::2], self.attn_pickle_dirs[1::2]) 
                    if model in _dir1 and model in _dir2
                ]

            else:
                assert False, f'Invalid component {compPath}'
            
            for _dir in _dirs:
                overlap, component_v, reals, nonce = conf_matrix_individual(topK, _dir)
                overall_comp.append(overlap)
                overall_comp_values.append(component_v)
                all_comp_overlap.append(overlap)
                all_comp_values.append(component_v)
        
            average_df = pd.DataFrame(columns=reals, index=nonce)
            
            for row in nonce:
                for col in reals:
                    average_df.loc[(average_df.index == row), col] = float(np.mean([overall_comp[i].loc[(overall_comp[i].index == row), col] for i in range(len(overall_comp))]))
            
            average_df = average_df.apply(pd.to_numeric)

            im = ax.imshow(average_df, cmap='binary', aspect='auto', interpolation='nearest')
            ax.set_yticks(range(len(sorted(average_df.index))), labels=[self.grammar_names[col] for col in average_df.index], fontsize=18, rotation=45)
            ax.set_title(f'{self.labels[model]} {compPath.title()}', fontsize=15)

            # -------------------------------- Draw lines -------------------------------------------------
            ax.axvline(x=2.5, color='black')
            # plt.axvline(x=ux_tick[0], color=unreal_color)
            ax.axhline(y=2.5, color='black')
            # plt.axhline(y=ux_tick[0], color=unreal_color)
        axes[-1].set_xticks(range(len(average_df.columns)), labels=[self.grammar_names[col] for col in average_df.columns], fontsize=18, rotation=-45, ha='left')
        axes[-2].set_xticks(range(len(average_df.columns)), labels=[self.grammar_names[col] for col in average_df.columns], fontsize=18, rotation=-45, ha='left')
        fig.colorbar(im, ax=axes[1], orientation='vertical', label='Overlap %')
        fig.colorbar(im, ax=axes[3], orientation='vertical', label='Overlap %')
        fig.colorbar(im, ax=axes[5], orientation='vertical', label='Overlap %')
        # fig.tight_layout(rect=[0, 0, 1, 1])
        fig.savefig(f'{self.op_dir}/{self.op_dir.split('/')[-1]}-{self.token_type}-{compPath}-overlap-{topK}.png')
        return all_comp_overlap, all_comp_values
    
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

    def get_stat_sig(self, reals, unreals, reals_unreals, component, lang):
        t_reals = [val[0] for val in reals]
        t_unreals = [val[0] for val in unreals]
        t_reals_unreals = [val[0] for val in reals_unreals]
        print('REALS ', len(t_reals), len(t_unreals), len(t_reals_unreals))
        print(f'############## {component}-{lang} ################')
        print('CxC vs NXN', stats.mannwhitneyu(t_reals, t_unreals))
        print('C-C vs C-N', stats.mannwhitneyu(t_reals, t_reals_unreals))
        print('N-N vs C-N', stats.mannwhitneyu(t_unreals, t_reals_unreals))

    def get_overlaps(self, conf_matrix, model_name):
        reals = { 'en': [], 'it': [], 'jp': [] }
        unreals = { 'en': [], 'it': [], 'jp': [] }
        reals_unreals = { 'en': [], 'it': [], 'jp': [] }
        cols = conf_matrix.index
        for ind in cols:
            for col in conf_matrix.columns:
                r_en = []
                u_en = []
                ru_en = []
                
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
                    reals_unreals['en'].append(ru_en)
                    
        t_reals = [val for key in reals.keys() for val_list in reals[key] for val in val_list]
        t_unreals = [val for key in reals.keys() for val_list in unreals[key]  for val in val_list]
        t_reals_unreals = [val for key in reals.keys() for val_list in reals_unreals[key]  for val in val_list]
        # print(f'############## {model_name} ################')
        # print('C-C vs N-N', stats.mannwhitneyu(t_reals, t_unreals))
        # print('C-C vs C-N', stats.mannwhitneyu(t_reals, t_reals_unreals))
        # print('N-N vs C-N', stats.mannwhitneyu(t_unreals, t_reals_unreals))
        return reals, unreals, reals_unreals
        
    def get_stat_sig(self, reals, unreals, reals_unreals, component, lang):
        real_vals = [val for key in reals.keys() for val_list in reals[key] for val in val_list]
        unreal_vals = [val for key in reals.keys() for val_list in unreals[key]  for val in val_list]
        real_unreal_vals = [val for key in reals.keys() for val_list in reals_unreals[key]  for val in val_list]
        print('REALS ', len(real_vals), len(unreal_vals), len(real_unreal_vals))
        print(f'############## {component} {lang} ################')
        print('C-C vs N-N', stats.mannwhitneyu(real_vals, unreal_vals))
        print('C-C vs C-N', stats.mannwhitneyu(real_vals, real_unreal_vals))
        print('N-N vs C-N', stats.mannwhitneyu(unreal_vals, real_unreal_vals))
        
        
    def draw_overlap_bars_plot(self):
        model_names = self.model_names
        img_prefix = 'across-models'
        m_avg_reals = {}
        m_avg_unreals = {}
        m_avg_reals_unreals = {}
        fig, axes = plt.subplots(1, len(model_names), figsize=(5 * len(model_names) + 1, 5), sharex=True, sharey=True)
        languages = []
        # Extract the categories and the values self.
        a_overlap, a_values = self.conf_matrix_plot("attn", 0.01)
        m_overlap, m_values = self.conf_matrix_plot("mlp", 0.01)
        ctr = 0
        for model, ax in zip(model_names, axes):
            a_reals, a_unreals, a_reals_unreals = self.get_overlaps(a_overlap[ctr], model)
            m_reals, m_unreals, m_reals_unreals = self.get_overlaps(m_overlap[ctr], model)
            reals = {}
            unreals = {}
            reals_unreals = {}    
        
            for lang in m_reals:
                if len(m_reals[lang]) > 0:
                    if not lang in m_avg_reals:
                        m_avg_reals[lang] = []
                        m_avg_unreals[lang] = []
                        m_avg_reals_unreals[lang] = []
                    reals[lang] = m_reals[lang] + a_reals[lang]
                    unreals[lang] = m_unreals[lang] + a_unreals[lang]
                    reals_unreals[lang] = m_reals_unreals[lang] + a_reals_unreals[lang]
                    
                    m_avg_reals[lang] += reals[lang]
                    m_avg_unreals[lang] += unreals[lang]
                    m_avg_reals_unreals[lang] += reals_unreals[lang]
            languages = list(reals.keys())
            
            reals_bar = [mean([val for item in reals[lang] for val in item]) for lang in languages]
            unreals_bar = [mean([val for item in unreals[lang] for val in item]) for lang in languages]
            reals_unreals_bar = [mean([val for item in reals_unreals[lang] for val in item]) for lang in languages]
            
            reals_var = [sem([val for item in reals[lang] for val in item]) for lang in languages]
            unreals_var = [sem([val for item in unreals[lang] for val in item]) for lang in languages]
            reals_unreals_var = [sem([val for item in reals_unreals[lang] for val in item]) for lang in languages]
            
            num_langs = len(languages)
            
            bar_width = 0.2
            index = np.arange(num_langs)
        
            ax.bar(index - bar_width, reals_bar, bar_width, label='H(ZZ x EN)', color=self.real_color, edgecolor='black')
            ax.errorbar(index - bar_width, reals_bar, yerr=reals_var, fmt='none', c='black', capsize=5)
            ax.bar(index, unreals_bar, bar_width, label='L(ZZ x EN)', color=self.unreal_color, edgecolor='black')
            ax.errorbar(index, unreals_bar, yerr=unreals_var, fmt='none', c='black', capsize=5)
            ax.bar(index + bar_width, reals_unreals_bar, bar_width, label='H(ZZ) x L(EN)', color=self.mixed_color, edgecolor='black')
            ax.errorbar(index + bar_width, reals_unreals_bar, yerr=reals_unreals_var, fmt='none', c='black', capsize=5)
            
            ax.set_title(self.labels[model])
            ax.set_xticks(index)
            ax.set_xticklabels([f'{key.upper()}, ZZ' for key in reals.keys()])
            plt.tight_layout()
            ctr += 1
        axes[0].set_yticks([0,0.25,0.5, 0.75,1])
        axes[0].set_ylabel(f'Mean Overlap \n Percentage')
        axes[3].legend(ncol=3, handlelength=1, handleheight=1, columnspacing=1, frameon=False, bbox_to_anchor=(0, 1.2), loc='upper center')
        fig.savefig(f'{self.op_dir}/{self.op_dir.split('/')[-1]}-{self.token_type}-mean-overlaps.png')
        
        for lang in m_avg_reals.keys():
            print(f'############## {self.token_type} ################')
            self.get_stat_sig(m_avg_reals[lang], m_avg_unreals[lang], m_avg_reals_unreals[lang], 'attn+mlp', lang)
        # Model average plot
        m_reals_bar = [mean([val for item in m_avg_reals[lang] for val in item]) for lang in languages]
        m_unreals_bar = [mean([val for item in m_avg_unreals[lang] for val in item]) for lang in languages]
        m_reals_unreals_bar = [mean([val for item in m_avg_reals_unreals[lang] for val in item]) for lang in languages]
        
        m_reals_var = [sem([val for item in m_avg_reals[lang] for val in item]) for lang in languages]
        m_unreals_var = [sem([val for item in m_avg_unreals[lang] for val in item]) for lang in languages]
        m_reals_unreals_var = [sem([val for item in m_avg_reals_unreals[lang] for val in item]) for lang in languages]
        
        num_langs = len(languages)
        
        with plt.rc_context({'font.size': 21}):
            bar_width = 0.15
            index = np.arange(num_langs)
            plt.figure(figsize=(7, 7))
            plt.bar(index - bar_width, m_reals_bar, bar_width, label='H(ZZ x EN)', color=self.real_color, edgecolor='black')
            plt.errorbar(index - bar_width, m_reals_bar, yerr=m_reals_var, fmt='none', c='black', capsize=5)
            plt.bar(index, m_unreals_bar, bar_width, label='L(ZZ x EN)', color=self.unreal_color, edgecolor='black')
            plt.errorbar(index, m_unreals_bar, yerr=m_unreals_var, fmt='none', c='black', capsize=5)
            plt.bar(index + bar_width, m_reals_unreals_bar, bar_width, label='H(ZZ) x L(EN)', color=self.mixed_color, edgecolor='black')
            plt.errorbar(index + bar_width, m_reals_unreals_bar, yerr=m_reals_unreals_var, fmt='none', c='black', capsize=5)
            # plt.legend(bbox_to_anchor=(0.5, 1.15), loc='lower center', ncol=3, frameon=False)
            plt.legend()
            plt.yticks([0, 0.25, 0.5, 0.75,1])
            plt.ylabel('Mean Overlap \n Percentage')
            plt.xticks(index, labels=[f'{key.upper()}, ZZ' for key in m_avg_reals.keys()])
            plt.tight_layout()
            plt.savefig(f'{self.op_dir}/{self.op_dir.split('/')[-1]}-model-mean-{self.token_type}-overlaps.png')
        

class Expt3:
    def __init__(self, config):
        self.get_files = config.get_files
        self.model_dirs = config.model_dirs_exp_3
        self.model_names = config.model_names
        self.token_type = config.token_type
        self.grammar_names = config.grammar_names
        self.constants = config.constants
        self.op_dir = config.op_dirs['expt3']
        self.expt1 = Expt1(config)
        if self.token_type == 'nonce_conv':
            self.grammar_names = { g for g in self.grammar_names if 'en_S-' in g}
        self.labels = config.constants.model_labels
    
    def create_plot(self):
        old_df = pd.DataFrame(columns=['model_name', 'acc', 'language', 'grammar', 'real']) 
        for model_name in self.model_names:
            expt1_df = self.expt1.create_df(model_name)
            old_df = pd.concat([old_df, expt1_df])
        
        def init_df(which):
            full_df = pd.DataFrame(columns=['model_name', 'acc', 'language', 'grammar', 'real'])
            for model_name in self.model_names:
                df = self.create_df(model_name, which)
                full_df = pd.concat([full_df, df])
            return full_df.reset_index()
        
        real_df = init_df('real')
        unreal_df = init_df('unreal')
        random_df = init_df('random')
        
        assert len(real_df) == len(unreal_df) == len(random_df), f'Unequal lengths, {len(real_df)}, {len(unreal_df)}, {len(random_df)} {self.token_type}'
        assert len(real_df) > 0
        
        assert list(real_df.columns) == list(unreal_df.columns) == list(random_df.columns), 'Columns do not match'
        assert list(real_df['language'].unique()) == list(unreal_df['language'].unique()) == list(random_df['language'].unique()), 'Languages do not match'
        assert list(real_df['grammar'].unique()) == list(unreal_df['grammar'].unique()) == list(random_df['grammar'].unique()), 'Grammars do not match'
        assert list(real_df['model_name'].unique()) == list(unreal_df['model_name'].unique()) == list(random_df['model_name'].unique()), 'Models do not match'
        assert list(real_df['real'].unique()) == list(unreal_df['real'].unique()) == list(random_df['real'].unique()), 'Real values do not match'
        unified_df = (
            random_df.merge(unreal_df, on=['model_name', 'language', 'grammar', 'real'], suffixes=('_random', '_unreal'))
                .merge(real_df, on=['model_name', 'language', 'grammar', 'real'])
                .merge(old_df, on=['model_name', 'language', 'grammar', 'real'], suffixes=('', '_old'))
        )
        
        assert len(unified_df) > 0, 'Unified DF is empty'

        self.unified_df = unified_df.rename(columns={
            "acc_random": "random_acc",
            "acc_unreal": "unreal_acc",
            "acc": "real_acc",
            "acc_old": "old_acc"
        })
        
        self.unified_df['real_diff'] = (self.unified_df['real_acc'] - self.unified_df['old_acc'])/self.unified_df['old_acc']
        self.unified_df['unreal_diff'] = (self.unified_df['unreal_acc'] - self.unified_df['old_acc'])/self.unified_df['old_acc']
        self.unified_df['random_diff'] = (self.unified_df['random_acc'] - self.unified_df['old_acc'])/self.unified_df['old_acc']
        
        self.create_plot_lang()    
        
    def create_plot_grammar_avg(self):
        plt.rcParams.update({'font.size': 21})
        grammar = self.grammar_names.keys()
        avg_df = self.unified_df[self.unified_df['grammar'].isin(grammar)].groupby('grammar')['acc'].mean()
        colors = []
        for grammar in avg_df.index:
            if '-r-' in grammar:
                colors.append(self.constants.real_color)
            else:
                colors.append(self.constants.unreal_color)

        avg_df = avg_df.sort_index(ascending=True)
        plt.figure(figsize=(12, 15))
        plt.bar(avg_df.index, avg_df.values, color=colors, edgecolor='black')
        plt.ylabel('Average Accuracy')
        plt.ylim(0, 1)

        # Creating legend handles
        real_patch = plt.Line2D([0], [0], color=self.constants.real_color, lw=4, label='H')
        unreal_patch = plt.Line2D([0], [0], color=self.constants.unreal_color, lw=4, label='L')

        plt.xticks(range(len(self.grammar_names.keys())), labels=self.grammar_names.values(), rotation=90)
        plt.legend(handles=[real_patch, unreal_patch], ncol=2)
        plt.tight_layout()
        plt.savefig(f'{self.op_dir}/{self.op_dir.split('/')[-1]}-{self.token_type}-average-grammars.png', dpi=300)
        plt.show()

    def create_plot_lang_avg(self):
        plt.rcParams.update({'font.size': 48})
        lang = self.full_model_df['language'].unique()
        r_data = []
        u_data = []

        for l in lang:
            r_data.append(self.full_model_df[(self.full_model_df['real'] == 1) & (self.full_model_df['language'] == l)]['acc'].tolist())
            u_data.append(self.full_model_df[(self.full_model_df['real'] == 0) & (self.full_model_df['language'] == l)]['acc'].tolist())

        r_means = [statistics.mean(rd) for rd in r_data]
        u_means = [statistics.mean(ud) for ud in u_data]
        r_var = [sem(rd) for rd in r_data]
        u_var = [sem(ud) for ud in u_data]

        fig, ax = plt.subplots(figsize=(20, 15))
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        bar_width = 0.2
        index = np.arange(len(lang))

        ax.bar(index, r_means, bar_width, label='H', color=[self.constants.real_color], edgecolor='black')
        ax.bar(index + bar_width, u_means, bar_width, label='L', color=[self.constants.unreal_color], edgecolor='black')

        plt.grid(True, which='major', axis='y', linestyle='--', linewidth=1.5, color='gray', alpha=1)
        plt.errorbar(index, r_means, yerr=r_var, fmt='none', c='black', capsize=5)
        plt.errorbar(index + bar_width, u_means, yerr=u_var, fmt='none', c='black', capsize=5)
        ax.set_ylabel('Mean Accuracy (Average)')
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(lang)
        ax.legend(frameon=False, ncol=2)
        plt.tight_layout()
        fig.savefig(f'{self.op_dir}/{self.op_dir.split('/')[-1]}-{self.token_type}-average-lang.png', dpi=300)
        plt.show()
    
    def create_plot_grammar(self, model_name):
        plt.rcParams.update({'font.size': 21})
        
        grammar = self.grammar_names.keys()
        unified_df = self.unified_df[self.unified_df['model_name'] == model_name].sort_values(by='grammar')
        
        h_data = []
        l_data = []
        r_data = []
        
        h_err = []
        l_err = []
        r_err = []
        # for g in grammar:
                
        plt.figure(figsize=(12, 15))
        accuracies = self.model_df[self.model_df['grammar'].isin(grammar)]['acc']
        plt.bar(sorted(self.grammar_names.values()), accuracies, color=colors, edgecolor='black')
        plt.ylabel('Accuracy')
        plt.ylim(0,1)

        # Creating legend handles
        real_patch = plt.Line2D([0], [0], color=self.constants.real_color, lw=4, label='H')
        unreal_patch = plt.Line2D([0], [0], color=self.constants.unreal_color, lw=4, label='L')

        plt.xticks(rotation=90)
        plt.legend(handles=[real_patch, unreal_patch], ncol=2)
        plt.tight_layout()
        plt.savefig(f'{self.op_dir}/{self.op_dir.split('/')[-1]}-{self.token_type}-{self.model_name}-grammars.png', dpi=300)
        plt.show()

    def stat_sig(self, avg):
        for l in avg:
            # ABLATION_grammar
            H_h = avg[l][0]
            H_l = avg[l][1]
            L_h = avg[l][2]
            L_l = avg[l][3]
            R_h = avg[l][4]
            R_l = avg[l][5]
            
            print(len(H_h), len(H_l), len(L_h), len(L_l), len(R_h), len(R_l))
            print(f'####################### {l} #######################')
            print('H_h vs L_h', stats.mannwhitneyu(H_h, L_h))
            print('H_h vs R_h', stats.mannwhitneyu(H_h, R_h))
            print('L_h vs R_h', stats.mannwhitneyu(L_h, R_h))
            
            print('H_l vs L_l', stats.mannwhitneyu(H_l, L_l))
            print('H_l vs R_l', stats.mannwhitneyu(H_l, R_l))
            print('L_l vs R_l', stats.mannwhitneyu(L_l, R_l))
        
        
    def create_plot_lang(self):
        # plt.rcParams.update({'font.size': 48})
        plt.rcParams.update({'font.size': 21})
        lang = self.unified_df['language'].unique()
        fig, axes = plt.subplots(1, len(self.model_names), figsize=(5 * len(self.model_names) + 1, 5), sharex=True, sharey=True)
        avg = {l: [] for l in lang}
        for ax, model in zip(axes, self.model_names):
            h_data = []
            l_data = []
            r_data = []
            h_err = []
            l_err = []
            r_err = []
            for l in lang:
                unified_df = self.unified_df[self.unified_df['model_name'] == model]
                h_real_ablate = unified_df[(unified_df['real'] == 1) & (unified_df['language'] == l)]['real_diff']
                h_unreal_ablate = unified_df[(unified_df['real'] == 1) & (unified_df['language'] == l)]['unreal_diff']
                h_random_ablate = unified_df[(unified_df['real'] == 1) & (unified_df['language'] == l)]['random_diff']
                
                l_real_ablate = unified_df[(unified_df['real'] == 0) & (unified_df['language'] == l)]['real_diff']
                l_unreal_ablate = unified_df[(unified_df['real'] == 0) & (unified_df['language'] == l)]['unreal_diff']
                l_random_ablate = unified_df[(unified_df['real'] == 0) & (unified_df['language'] == l)]['random_diff']
                
                h_data += [mean(h_real_ablate), mean(l_real_ablate)]
                l_data += [mean(h_unreal_ablate), mean(l_unreal_ablate)]
                r_data += [mean(h_random_ablate), mean(l_random_ablate)]
                
                h_err += [sem(h_real_ablate), sem(l_real_ablate)]
                l_err += [sem(l_unreal_ablate), sem(l_unreal_ablate)]
                r_err += [sem(h_random_ablate), sem(l_random_ablate)]
            
                for idx, data in enumerate([h_real_ablate, l_real_ablate, h_unreal_ablate, l_unreal_ablate, h_random_ablate, l_random_ablate]):
                    if idx == len(avg[l]):
                        avg[l].append(list(data))
                    else:
                        avg[l][idx] += list(data)
            
            n = len(lang) * 2

            bar_width = 0.2
            index = np.arange(n)
            x_labels = []
            for l in lang:
                if n == 2:
                    x_labels.append(f'ZZ-H')
                    x_labels.append(f'ZZ-L')
                else:
                    x_labels.append(f'{l.upper()}-H')
                    x_labels.append(f'{l.upper()}-L')
            # print(index, h_data, model, self.token_type)
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)
            ax.bar(index, h_data, bar_width, label='H', color=[self.constants.real_color], edgecolor='black')
            ax.bar(index + bar_width, l_data, bar_width, label='L', color=[self.constants.unreal_color], edgecolor='black')
            ax.bar(index + 2*bar_width, r_data, bar_width, label='Random', color=['gray'], edgecolor='black')
            ax.errorbar(index, y=h_data, yerr=h_err, fmt='none', c='black', capsize=5)
            ax.errorbar(index + bar_width, y=l_data, yerr=l_err, fmt='none', c='black', capsize=5)
            ax.errorbar(index + 2*bar_width, y=r_data, yerr=r_err, fmt='none', c='black', capsize=5)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.set_xticks(index + bar_width/3, labels=x_labels, rotation=90)
            ax.set_title(self.labels[model])
        axes[0].set_ylabel('Mean relative change\n in accuracy')
        axes[-1].legend(frameon=False, ncol=1, title="Ablations: ", loc='upper left', bbox_to_anchor=(1, 1))
        # axes[-1].legend(ncol=1, handlelength=1, handleheight=1, columnspacing=1, frameon=False, bbox_to_anchor=(1, 1), loc='upper left')
        fig.tight_layout(rect=[0, 0, 1, 1])
        fig.savefig(f'{self.op_dir}/{self.op_dir.split('/')[-1]}-{self.token_type}-lang.png', dpi=300)
        
        
        n = len(lang) * 2

        h_data = []
        l_data = []
        r_data = []
        
        h_err = []
        l_err = []
        r_err = []
        
        self.stat_sig(avg)
        for l in avg:
            h_data += [mean(avg[l][0]), mean(avg[l][1])]
            l_data += [mean(avg[l][2]), mean(avg[l][3])]
            r_data += [mean(avg[l][4]), mean(avg[l][5])]
            
            h_err += [sem(avg[l][0]), sem(avg[l][1])]
            l_err += [sem(avg[l][2]), sem(avg[l][3])]
            r_err += [sem(avg[l][4]), sem(avg[l][5])]
        
        bar_width = 0.2
        index = np.arange(n)
        x_labels = []
        for l in lang:
            if n == 2:
                x_labels.append(f'ZZ-H')
                x_labels.append(f'ZZ-L')
            else:
                x_labels.append(f'{l.upper()}-H')
                x_labels.append(f'{l.upper()}-L')

        plt.rcParams.update({'font.size': 15})
        if self.token_type == 'nonce':
            plt.figure(figsize=(5, 5))
            plt.bar(index, h_data, bar_width, label='ZZ (H)', color=[self.constants.real_color], edgecolor='black')
            plt.bar(index + bar_width, l_data, bar_width, label='ZZ (L)', color=[self.constants.unreal_color], edgecolor='black')
            plt.bar(index + 2*bar_width, r_data, bar_width, label='ZZ (Random)', color=['gray'], edgecolor='black')
        elif self.token_type == 'nonce_conv':
            plt.figure(figsize=(5, 5))
            plt.bar(index, h_data, bar_width, label='EN (H)', color=[self.constants.real_color], edgecolor='black')
            plt.bar(index + bar_width, l_data, bar_width, label='EN (L)', color=[self.constants.unreal_color], edgecolor='black')
            plt.bar(index + 2*bar_width, r_data, bar_width, label='EN (Random)', color=['gray'], edgecolor='black')
        else:
            plt.figure(figsize=(5, 3))
            plt.bar(index, h_data, bar_width, label='(H)', color=[self.constants.real_color], edgecolor='black')
            plt.bar(index + bar_width, l_data, bar_width, label='(L)', color=[self.constants.unreal_color], edgecolor='black')
            plt.bar(index + 2*bar_width, r_data, bar_width, label='(Random)', color=['gray'], edgecolor='black')
        plt.errorbar(index, y=h_data, yerr=h_err, fmt='none', c='black', capsize=5)
        plt.errorbar(index + bar_width, y=l_data, yerr=l_err, fmt='none', c='black', capsize=5)
        plt.errorbar(index + 2*bar_width, y=r_data, yerr=r_err, fmt='none', c='black', capsize=5)
        plt.xticks(index + bar_width/3, labels=x_labels)
        
        # plt.suptitle("Average ablations \n across models")
        plt.ylabel('Mean relative change \n in accuracy')
        # if not self.token_type == 'nonce_conv':
        plt.legend(frameon=False, ncol=3, title="Ablations: ", loc='lower center', bbox_to_anchor=(0.4, 1.05), fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 1])
        plt.savefig(f'{self.op_dir}/{self.op_dir.split('/')[-1]}-{self.token_type}-avg-lang.png', dpi=300)
    
    def create_df(self, model_name, which):
        model_df = []
        # print("WHICH ", model_name, self.model_dirs)
        for m_dir in self.model_dirs[which]:
            if not model_name in m_dir:
                continue
            model_df.append(self.get_files(m_dir))
        model_df = pd.concat(model_df)
        model_df['model_name'] = model_name
        model_df = model_df.reset_index()
        
        assert len(model_df) > 0, f"Model {model_name} not found in {which} data"
        return model_df
        
    def create_all_models_same_plot(self):
        plt.rcParams.update({'font.size': 21})
        markers = ['o', 's', '^', 'D', 'p', '*', '<', '>', 'h']  # Different marker styles
        colors = ['green', 'orange', 'purple', 'brown', 'teal', 'magenta']
        # Get the unique languages
        if self.token_type == 'conventional':
            languages = self.full_model_df['language'].unique()
        else:
            languages = ['EN']

        # Create subplots: one subplot per language
        fig, axes = plt.subplots(1, len(languages), figsize=(5 * len(languages) + 1, 5), sharex=True, sharey=True)
        # If only one subplot (1 language), convert axes to a list for iteration
        if len(languages) == 1:
            axes = [axes]

        handles, labels = [], []
        # Iterate through each language and plot
        for ax, lang in zip(axes, languages):
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)
            ax.set_xlim(0, 1.1)
            ax.set_ylim(0, 1.1)
            ax.plot([0, 1.1], [0, 1.1], linestyle='--', color='gray', linewidth=1)
            ax.set_xticks(np.arange(0, 1.1, 0.25))
            ax.set_yticks(np.arange(0, 1.1, 0.25))
            ax.set_title(f'{lang}')

            for i in range(len(self.model_names)):
                x = self.full_model_df[
                    (self.full_model_df['language'] == lang) &
                    (self.full_model_df['model_name'] == self.model_names[i]) & 
                    (self.full_model_df['grammar'].isin(self.grammar_names.keys())) &
                    (self.full_model_df['real'] == 1)
                ]['acc']
                y = self.full_model_df[
                    (self.full_model_df['language'] == lang) &
                    (self.full_model_df['model_name'] == self.model_names[i]) & 
                    (self.full_model_df['grammar'].isin(self.grammar_names.keys())) &
                    (self.full_model_df['real'] == 0)
                ]['acc']
                scatter = ax.scatter(
                    x, y, 
                    marker=markers[i % len(markers)], 
                    color=colors[i % len(colors)], 
                    s=100, alpha=0.7, label=self.model_names[i]
                )
                # Append handles and labels for legend
                if len(handles) < len(self.model_names):  # Ensure only unique labels
                    handles.append(scatter)
                    labels.append(self.model_names[i])

            ax.set_xlabel('H')
            ax.set_ylabel('L')
        if self.token_type == 'conventional':
            axes[2].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=15, frameon=False, ncol=1)
        else:
            axes[0].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=15, frameon=False, ncol=1)
        plt.tight_layout(rect=[0, 0, 1, 0.9])  # Leave space for the legend above the plots

        # Save the figure
        plt.savefig(f'{self.op_dir}/{self.op_dir.split('/')[-1]}-{self.token_type}-all-models-accuracies.png', dpi=300)
        plt.close()
        
def main():
    MODEL_NAMES = ['llama-2-7b', 'llama-3.1-8b', 'llama-3.1-70b', 'qwen-2-0.5b', 'qwen-2-1.5b', 'mistral-v0.3']
    TOKEN_TYPES = ['nonce', 'conventional', 'nonce_conv']
    
    for token_type in TOKEN_TYPES:
        if not token_type == 'nonce_conv':
            # config = Config(MODEL_NAMES, token_type, 1)
            # expt_1 = Expt1(config)
            # expt_1.create_plot()
        
            # config = Config(MODEL_NAMES, token_type, 1)
            # expt_2 = Expt2(config)
            # expt_2.create_plot()

            config = Config(MODEL_NAMES, token_type, 3)
            expt_3 = Expt3(config)
            expt_3.create_plot()
        
        if token_type == 'nonce_conv':
        #     # config = Config(MODEL_NAMES, token_type, 2)
        #     # expt_2_1 = Expt2_1_nonce_conv(config)
        #     # expt_2_1.create_plot()
            
            config = Config(MODEL_NAMES, token_type, 3)
            expt_3 = Expt3(config)
            expt_3.create_plot()
if __name__ == "__main__":
    main()
    