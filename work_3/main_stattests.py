import itertools
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

# Read results pickles
dataset = datasets[id_data]
path_read = f'results/{dataset}/'
with open(f'{path_read}pd_results_sum.pickle', 'rb') as file:
    pd_results_sum = pickle.load(file)
with open(f'{path_read}dic_results.pickle', 'rb') as file:
    dic_results = pickle.load(file)

# See and save results summary
print('0. Summary of experiment results')
print(pd_results_sum)

## Step 1. Best model and models nos statistically different with Wilcoxon
# Sort from best to worst mean_sc
sorted = pd_results_sum.sort_values('mean_sc', ascending=False).reset_index()
# Select cases with highest accuracy, base for first step comparison (there may be many sharing first place)
sorted_filter = sorted[sorted['mean_sc'] == sorted['mean_sc'].max()]
# Wilcoxon between all possible pairs to make first filter (all are done to consult them easily and compare to nemenyi)
matrix_pvalues, matrix_diff_mean_acc = wilcoxon_experiments(dic_results)
df_matrix_pvalues = pd.DataFrame(matrix_pvalues, columns=pd_results_sum['ex'], index=pd_results_sum['ex'])
# Get these experiments and those with no significant difference with them, as possible candidates
p_values_cand = df_matrix_pvalues.loc[sorted_filter['ex'], :] # Rows for best
any_case = (p_values_cand > 0.05).any()  # Experiments with at least one case with null hypothesis
any_case = p_values_cand.columns[any_case].tolist()
sorted_candidates = sorted[sorted['ex'].isin(any_case)]
print(f'Step 1 candidates ({sorted_candidates.shape[0]}):')
print(sorted_candidates)

## Step 2. Get best models for each parameter variation
# Get best model for each of the parameter variations
df = pd_results_sum
idx = df.groupby(['k']).transform(max) == df['mean_sc']
params = [['k'], ['distance', 'mink_power'], ['decision'], ['weighting']]
sel = []
for p in params:
    df_i = df.sort_values('mean_sc', ascending=False).drop_duplicates(p)
    df_i['base_parameter'] = p[0]
    sel.append(df_i)
sel = pd.concat(sel)
sel_sorted = sel.sort_values('mean_sc', ascending=False).reset_index()
sel_sorted.to_csv('results/bestmodel_parametrs.csv')
print(sel_sorted)
sel_exp = sel_sorted.drop_duplicates(subset=['ex'])
list_sel = sel_exp['ex'].tolist()

# Compare candidates with Friedman Chi square test, and if rejecting Ho, estimate Nemenyi P-Values between pairs
nemenyi_pvalues = friedman_experiments(dic_results, list_sel, chi_pvalue_threshold=0.05)
if nemenyi_pvalues is None:
    print('No Nemenyi required as Friedman is not statistically significant')
else:
    print('Nemenyi P-Values matrix:')
    print(nemenyi_pvalues)
    plot_nemenyi(nemenyi_pvalues)  # Plot Nemenyi heatmap
    plt.savefig('results/heatmap_Nemenyi.png')

    nem_diff = nemenyi_pvalues[nemenyi_pvalues == 0.001].dropna(axis=0, how='all').dropna(axis=1, how='all')
    print(nem_diff)


# Exploratory: Heatmaps and
pd_results_sum.to_csv('results/summary_experiments.csv')
# df_matrix_diff_acc = pd.DataFrame(matrix_diff_mean_acc, columns=pd_results_sum['ex'], index=pd_results_sum['ex'])
# df_matrix_diff_acc.to_csv('results/matrix_mean_acc.csv')
sorted.to_csv('results/sorted_all_exp.csv')
sorted_candidates.to_csv('results/sorted_candidates_wilcoxon.csv')
# df_matrix_pvalues.to_csv('results/matrix_pvalues.csv')
# heatmap(matrix=matrix_pvalues, title='Heatmap of experiments Wilcoxon P-Values', path='results/heatmap_wilcoxon.png')

