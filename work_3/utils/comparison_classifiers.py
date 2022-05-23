import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import friedmanchisquare, wilcoxon
from scikit_posthocs import posthoc_nemenyi_friedman, sign_plot
import pdb

def friedman_experiments(dic_results, candidates, chi_pvalue_threshold):
    # Extract accuracies together with experiment id
    list_ex = candidates #[*dic_results.keys()]
    num_ex = len(list_ex)
    l_acc_sc = [ex['acc_sc'] for name_ex, ex in dic_results.items() if name_ex in list_ex]
    np_acc_sc = np.array(l_acc_sc)
    stat_chi, pvalue_chi = friedmanchisquare(*l_acc_sc)
    if pvalue_chi > chi_pvalue_threshold:
        print(f'Friedmann Chi-square with p-value of {pvalue_chi}:'
              f' Paired sample distributions are equal, not additional analysis required')
        nemenyi_pvalues = None
    else:
        print(f'Friedmann Chi-square with p-value of {pvalue_chi}:'
              ' Paired sample distributions are statistically different, Nemenyi pair analysis done')
        nemenyi_pvalues = posthoc_nemenyi_friedman(np_acc_sc.T)
        nemenyi_pvalues.set_axis(list_ex, axis='columns', inplace=True)
        nemenyi_pvalues.set_axis(list_ex, axis='rows', inplace=True)

    return nemenyi_pvalues


def plot_nemenyi(nemenyi_pvalues):
    heatmap_args = {'linewidths': 0.25, 'linecolor': '0.5', 'square': True,
                    'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}

    sign_plot(nemenyi_pvalues, **heatmap_args)


def wilcoxon_experiments(dic_results):
    # Extract accuracies together with experiment id
    list_ex = [*dic_results.keys()]
    num_ex = len(list_ex)
    np_matrix_pvalue = np.ones((num_ex, num_ex))
    np_matrix_diff_mean_acc = np.zeros((num_ex, num_ex))

    comb_ex = combinations(list_ex, 2)
    for ex_i, ex_j in comb_ex:
        # Get accuracies list from each experiment (one per fold)
        acc_i = dic_results[ex_i]['acc_sc']
        acc_j = dic_results[ex_j]['acc_sc']
        stat, pvalue = wilcoxon(x=acc_i, y=acc_j, zero_method='zsplit')
        # Save pvalue in matrix
        n_i = int(ex_i.split('ex_')[1])
        n_j = int(ex_j.split('ex_')[1])
        np_matrix_pvalue[n_i, n_j] = pvalue
        np_matrix_pvalue[n_j, n_i] = pvalue

        # Similar process but this time to save the difference between mean averages
        mean_sc_i = dic_results[ex_i]['mean_sc']
        mean_sc_j = dic_results[ex_j]['mean_sc']
        diff = mean_sc_i - mean_sc_j
        np_matrix_diff_mean_acc[n_i, n_j] = diff
        np_matrix_diff_mean_acc[n_j, n_i] = - diff # Always difference estimated ex1 - ex2

    return np_matrix_pvalue, np_matrix_diff_mean_acc

def friedman_experiments2(l_acc_sc, list_ex,chi_pvalue_threshold):
    np_acc_sc = np.array(l_acc_sc)
    stat_chi, pvalue_chi = friedmanchisquare(*l_acc_sc)
    if pvalue_chi > chi_pvalue_threshold:
        print(f'Friedmann Chi-square with p-value of {pvalue_chi}:'
              f' Paired sample distributions are equal, not additional analysis required')
        nemenyi_pvalues = None
    else:
        print(f'Friedmann Chi-square with p-value of {pvalue_chi}:'
              ' Paired sample distributions are statistically different, Nemenyi pair analysis done')
        nemenyi_pvalues = posthoc_nemenyi_friedman(np_acc_sc.T)
        nemenyi_pvalues.set_axis(list_ex, axis='columns', inplace=True)
        nemenyi_pvalues.set_axis(list_ex, axis='rows', inplace=True)

    return nemenyi_pvalues