import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.comparison_classifiers import *
from utils.visualizations import *
import pdb

# Import pickles with experiment results
# Parameters of execution
id_data = 0  # Select the desired data set with a value of 0, 1 or 2
datasets = ['pen-based', 'sick']
reductions = ['IB2', 'MENN']
best_base = {'pen-based': 'ex_0', 'sick': 'ex_23'}

df_data_reductions = {}
df_data_base = {}

folds_reductions = {}
folds_base = {}
for dataset in datasets:
    for reduction in reductions:
        path_read = f'results/{dataset}/reduction_{reduction}/'
        with open(f'{path_read}pd_results_sum_reduction.pickle', 'rb') as file:
            df_data_reductions[(dataset, reduction)] = pickle.load(file)
        with open(f'{path_read}dic_results_reduction.pickle', 'rb') as file:
            folds_reductions[(dataset, reduction)] = pickle.load(file)
    path_read = f'results/{dataset}/'
    with open(f'{path_read}pd_results_sum.pickle', 'rb') as file:
        df_data_base[dataset] = pickle.load(file)
    with open(f'{path_read}dic_results.pickle', 'rb') as file:
        folds_base[dataset] = pickle.load(file)


dic_df = {}
dic_acc_sc = {}
l_reductions = {'pen-based': list(), 'sick': list()}
for dataset in datasets:
    dic_df[dataset] = df_data_base[dataset][df_data_base[dataset]['ex'] == best_base[dataset]] # Get best model exp df
    dic_df[dataset]['ex'] = 'base'
    
    dic_acc_sc[(dataset, 'base')] = folds_base[dataset][best_base[dataset]]['acc_sc'] # folds from best model exp

    for reduction in reductions:
        df_i = df_data_reductions[(dataset, reduction)]
        df_i['ex'] = reduction
        dic_df[dataset] = dic_df[dataset].append(df_i)
        
        dic_acc_sc[(dataset, reduction)] = folds_reductions[(dataset, reduction)]['ex_0']['acc_sc']

dataset = 'pen-based'
l_cases = [dic_acc_sc[(dataset, 'base')], dic_acc_sc[(dataset, 'IB2')], dic_acc_sc[(dataset, 'MENN')]]



nemenyi_pvalues = friedman_experiments2(l_cases, ['base', 'IB2', 'MENN'], 0.05)

if nemenyi_pvalues is None:
    print('No Nemenyi required as Friedman is not statistically significant')
else:
    print('Nemenyi P-Values matrix:')
    print(nemenyi_pvalues)
    plot_nemenyi(nemenyi_pvalues)  # Plot Nemenyi heatmap
    plt.savefig('results/heatmap_Nemenyi.png')

    nem_diff = nemenyi_pvalues[nemenyi_pvalues == 0.001].dropna(axis=0, how='all').dropna(axis=1, how='all')
    print(nem_diff)


dic_df['pen-based'].to_csv('results/pen_based_reduction.csv')
dic_df['sick'].to_csv('results/sick_reduction.csv')
