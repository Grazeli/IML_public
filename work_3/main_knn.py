import itertools
import pickle

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from data_preprocessing import *
from utils.KNN import *
from utils.comparison_classifiers import *
from utils.visualizations import *
from sklearn.metrics import accuracy_score

# Parameters of execution
id_data = 0  # Select the desired data set with a value of 0, 1 or 2
datasets = ['pen-based', 'sick']  # https://datahub.io/machine-learning/sick   https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits
l_y_var = ['a17', 'Class']
nfolds = 10


# Select dataset and y_var
dataset = datasets[id_data]
y_var = l_y_var[id_data]
# Read and preprocess data
path_datasets = 'datasetsCBR/'
# Unify folds with a column with its corresponding fold
dic_df = load_arff_data_fold(path_datasets, dataset, nfolds)  # Dictionary with train and test df appended
# Generic function to preprocess data
dic_df = preprocessing(dic_df, y_var)

# Process to be repeated for each of the folds
k_vals = [1, 3, 5, 7]
to_tune_mink = {
    'k': k_vals,
    'distance': ['minkowski'],
    'mink_power': [1, 2],
    'decision': list(kNN.decision_func.keys()),
    'weighting': list(kNN.weight_func.keys()),
}
to_tune_cos = {
    'k': k_vals,
    'distance': ['cosine'],
    'mink_power': [None],
    'decision': list(kNN.decision_func.keys()),
    'weighting': list(kNN.weight_func.keys()),
}


def combinations(d):
    keys, values = zip(*d.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


param_list = combinations(to_tune_mink) + combinations(to_tune_cos)
dic_results = {}  # (parameters, fold, accuracy, efficiency) as keys
pd_results_sum = pd.DataFrame(columns=['ex'] + [*param_list[0]] + ['mean_sc', 'std_sc'])
for i, param in enumerate(param_list):
    acc_preds = []
    acc_sc = []
    time_folds = []
    ex_i = 'ex_{}'.format(i)
    print(f'{ex_i}/{len(param_list)}')
    print(param)
    for fold in range(nfolds):
        print('fold {}/{}'.format(fold+1, nfolds))
        dic_df_fold = {}
        # Select corresponding data for this fold
        for data_type in ['train', 'test']:
            df = dic_df[data_type]
            df = df[df['fold'] == fold]
            # Remove fold column
            dic_df_fold[data_type] = df.drop(columns=['fold'])

        # Main process
        X_train = dic_df_fold['train']
        y_train = X_train[y_var]
        X_train.drop(columns=[y_var], inplace=True)

        X_test = dic_df_fold['test']
        y_test = X_test[y_var]
        X_test.drop(columns=[y_var], inplace=True)

        clf = kNN(k=param['k'], distance=param['distance'], mink_power=param['mink_power'],
                  decision=param['decision'], weighting=param['weighting'])
        clf.fit(X_train, y_train)

        start_time = time.time()
        y_pred = clf.predict(X_test)
        end_time = time.time()

        k_time = end_time - start_time
        time_folds.append(k_time)
        sc = accuracy_score(y_test, y_pred)
        acc_preds.append(y_pred)
        acc_sc.append(sc)
    dic_results[ex_i] = {
        'param': param,
        'acc_preds': acc_preds,
        'acc_sc': acc_sc,
        'mean_sc': np.mean(acc_sc),
        'std_sc': np.std(acc_sc),
        'mean_time': np.mean(time_folds),
        'std_time': np.std(time_folds),
    }
    # Save summary, one row per experiment
    row_append = param
    row_append.update({'ex': ex_i,
                       'mean_sc': np.mean(acc_sc),
                       'std_sc': np.std(acc_sc),
                      'mean_time': np.mean(time_folds),
                      'std_time': np.std(time_folds)
                       })
    pd_results_sum = pd_results_sum.append(row_append, ignore_index=True)

# Save pickle dictionary
pkl_pd_results_sum = open('results/pd_results_sum.pickle', 'wb')
pickle.dump(pd_results_sum, pkl_pd_results_sum)
pkl_dic_results = open('results/dic_results.pickle', 'wb')
pickle.dump(dic_results, pkl_dic_results)

