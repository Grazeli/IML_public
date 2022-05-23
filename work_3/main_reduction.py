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
from utils.reduction import MENN, IB3, IB2
from utils.FCNN import *

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

reduction_func = {
    'MENN': MENN,
    'IB3': IB3,
    'IB2': IB2,
    'FCNN': FCNN,
}
# TODO change to chosen params
dataset_params = {
    'pen-based': {
        'k': [1],
        'distance': ['minkowski'],
        'mink_power': [1],
        'decision': ['majority'],
        'weighting': ['equal'],
        'reduction': [#None,
                      'IB3'],  # 'MENN' 'IB3'
    },
    'sick': {
        'k': [3],
        'distance': ['minkowski'],
        'mink_power': [1],
        'decision': ['IDW'],
        'weighting': ['IG'],
        'reduction': [#None,
                      'IB2'],  # 'MENN' 'IB3'
    },
}

params = dataset_params[datasets[id_data]]


def combinations(d):
    keys, values = zip(*d.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


param_list = combinations(params)
dic_results = {}  # (parameters, fold, accuracy, efficiency) as keys
pd_results_sum = pd.DataFrame(columns=['ex'] + [*param_list[0]] + ['mean_sc', 'std_sc'])
for i, param in enumerate(param_list):
    acc_preds = []
    acc_sc = []
    time_folds = []
    rows_folds = []
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
            if data_type == 'train':
                df_train = df.copy()
            # Remove fold column
            dic_df_fold[data_type] = df.drop(columns=['fold'])

        # Main process
        X_train = dic_df_fold['train']
        y_train = X_train[y_var]
        X_train.drop(columns=[y_var], inplace=True)

        X_test = dic_df_fold['test']
        y_test = X_test[y_var]
        X_test.drop(columns=[y_var], inplace=True)

        if param['reduction']:
            if param['reduction'] == 'FCNN':
                reduction = reduction_func[param['reduction']]
                clf = reduction([y_var, 'fold'])
                shape_X_prev = X_train.shape[0]
                Xy = clf.transform(df_train, y_var, param['k'])
                y_train = Xy[y_var]
                X_train = Xy.drop(columns=[y_var])
            else:
                reduction = reduction_func[param['reduction']]
                clf = reduction()
                shape_X_prev = X_train.shape[0]
                X_train, y_train = clf.fit_transform(X_train, y_train)

        clf = kNN(k=param['k'], distance=param['distance'], mink_power=param['mink_power'],
                  decision=param['decision'], weighting=param['weighting'])
        clf.fit(X_train, y_train)

        start_time = time.time()
        y_pred = clf.predict(X_test)
        end_time = time.time()

        k_time = end_time - start_time
        time_folds.append(k_time)
        print(f'{X_train.shape[0]/shape_X_prev}')
        rows_folds.append(X_train.shape[0]/shape_X_prev)
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
        'mean_rows': np.mean(rows_folds),
        'std_rows': np.std(rows_folds),
    }
    # Save summary, one row per experiment
    row_append = param
    row_append.update({'ex': ex_i,
                       'mean_sc': np.mean(acc_sc),
                       'std_sc': np.std(acc_sc),
                      'mean_time': np.mean(time_folds),
                      'std_time': np.std(time_folds),
                      'mean_rows': np.mean(rows_folds),
                      'std_rows': np.std(rows_folds),
                       })
    pd_results_sum = pd_results_sum.append(row_append, ignore_index=True)

# Save pickle dictionary
pkl_pd_results_sum = open('results/pd_results_sum_reduction.pickle', 'wb')
pickle.dump(pd_results_sum, pkl_pd_results_sum)
pkl_dic_results = open('results/dic_results_reduction.pickle', 'wb')
pickle.dump(dic_results, pkl_dic_results)

