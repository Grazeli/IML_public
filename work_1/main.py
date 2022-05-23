import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from utils.utils import *
from utils.KMeans import *
from utils.BisectingKMeans import *
from utils.FuzzyCMeans import *
from data_preprocessing import *
import itertools
import math

# Parameters of execution
id_data = 2  # Select the desired data set
datasets = ['vote.arff', 'breast-w.arff', 'heart-c.arff']
y_var = ['Class', 'Class', 'num']

# Read and preprocess data
path_datasets = 'datasets/'
dataset_path = path_datasets + datasets[id_data]
df = load_arff_data(dataset_path)
df_X, df_y = preprocessing(df, datasets[id_data], y_var[id_data])
df = df_X.copy()

# Real categories
df_complete = df_X.join(df_y)
pca_visualization(df_complete, df_y.columns[0], datasets[id_data][:-5], 'Real Categories')  # Plot over first two components

# Algorithm parameters
algorithms = {
    'DBSCAN': {
        'init': {
            'func': DBSCAN,
            'params': {
                'eps': np.arange(1.0, 3.0, 0.5),
                'metric': ['cosine', 'euclidean', 'manhattan'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            },
        },
        'fit': {
            'func': 'fit',
            'params': {},
        },
        'tune_on': 'init',
        'labels': 'labels_'
    },
    'KMeans': {
        'init': {
            'func': KMeans,
            'params': {
                'method': 'kmeans'
            },
        },
        'fit': {
            'func': 'fit',
            'params': {
                'n_clusters': np.arange(2, 8, 1),
            },
        },
        'tune_on': 'fit',
        'labels': 0
    },
    'KMedians': {
        'init': {
            'func': KMeans,
            'params': {
                'method': 'kmedians'
            },
        },
        'fit': {
            'func': 'fit',
            'params': {
                'n_clusters': np.arange(2, 8, 1),
            },
        },
        'tune_on': 'fit',
        'labels': 0
    },
    'KMeans++': {
        'init': {
            'func': KMeans,
            'params': {
                'method': 'kmeans++',
            },
        },
        'fit': {
            'func': 'fit',
            'params': {
                'n_clusters': np.arange(2, 8, 1),
            },
        },
        'tune_on': 'fit',
        'labels': 0
    },
    'Bisecting KMeans': {
        'init': {
            'func': BisectingKMeans,
            'params': {
                'method': 'kmeans',
            },
        },
        'fit': {
            'func': 'fit',
            'params': {
                'n_clusters': np.arange(2, 8, 1),
            },
        },
        'tune_on': 'fit',
        'labels': None
    },
    'Fuzzy C-Means': {
        'init': {
            'func': FuzzyCMeans,
            'params': {},
        },
        'fit': {
            'func': 'fit',
            'params': {
                'n_clusters': np.arange(2, 8, 1),
                'threshold': 0.1**np.arange(3, 6),
                'crisp': [True],
            },
        },
        'tune_on': 'fit',
        'labels': 1
    },
}

# Graph to find epsilon, not used for the parameter selection
graph_epsilon(df_X, datasets[id_data][:-5], 'Distances for NearestNeighbors')

# Execution of the algorithms
results = {}
for key, conf in algorithms.items():

    # Generate all combinations of parameter values
    to_tune = conf[conf['tune_on']]['params']
    keys, values = zip(*to_tune.items())
    tune_perm = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Delete invalid parameter combinations
    if key == 'DBSCAN':
        tune_perm = [t for t in tune_perm if not (
                t['metric'] == 'cosine' and
                t['algorithm'] in ['ball_tree', 'kd_tree'])]

    # Empty dataframe to store the results
    init_keys = list(conf['init']['params'].keys())
    fit_keys = list(conf['fit']['params'].keys())
    results[key] = pd.DataFrame(columns=init_keys + fit_keys)

    # Iterate on configurations from parameter combinations
    for i, params in enumerate(tune_perm):
        init_params = conf['init']['params']
        fit_params = conf['fit']['params']

        # Check which method receives the tuned parameters
        if conf['tune_on'] == 'init':
            init_params = params
        else:
            fit_params = params

        # Initialize model
        init = conf['init']['func'](**init_params)
        # Fit model
        fit = getattr(init, conf['fit']['func'])
        res = fit(df_X, **fit_params)
        pred_labels = res

        # Unpack the labels depending on the format received
        if isinstance(conf['labels'], int):
            pred_labels = res[conf['labels']]
        elif isinstance(conf['labels'], str):
            pred_labels = getattr(res, conf['labels'])

        # Score results and add them to the DataFrame
        scores = compute_scores(df_X, df_y.T.values[0], pred_labels)
        results[key] = results[key].append({'name': '{}_{}'.format(key, i),
                                            **init_params, **fit_params,
                                            **scores,
                                            'labels': pred_labels},
                                           ignore_index=True)
        # Save the visualization
        df = df_X.copy()
        df['labels'] = pred_labels
        pca_visualization(df, 'labels', datasets[id_data][:-5], '{}_{}'.format(key, i))

# Save tables with all configurations and results, in both csv and tex
for key, value in results.items():
    del value['labels']
    file = 'results/{}/{}_{}'.format(datasets[id_data][:-5],
                                     datasets[id_data][:-5], key)
    value.to_csv('{}.csv'.format(file))
    with open('{}.tex'.format(file), 'w') as tf:
        tf.write(value.to_latex())

# Empty DataFrame to store the best models selected
# by the metrics that do not use ground truth
score_cols = [col for col in results[key].columns if 'score' in col]
scores = {'silhouette_score': {
    'method': 'argmax', # We want to maximize it
    'df': pd.DataFrame(columns=score_cols + ['method'])
},
    'davies_bouldin_score': {
    'method': 'argmin', # We want to minimize it
    'df': pd.DataFrame(columns=score_cols + ['method'])
}}

# Plots for the selected models
for sc_key, sc_value in scores.items():
    # Generate dataframe with the results
    for key, df in results.items():
        names = df['name']
        df = df[score_cols]
        row = getattr(pd.to_numeric(df[sc_key]), sc_value['method'])()
        conf = names.iloc[row]
        sc_value['df'] = sc_value['df'].append({**(df.iloc[row].to_dict()),
                                                'method': key,
                                                'conf': conf},
                                                ignore_index=True)

    # Save in tex for easier access when checking values
    file = 'results/{}/{}_{}'.format(datasets[id_data][:-5],
                                     datasets[id_data][:-5], sc_key)
    with open('{}.tex'.format(file), 'w') as tf:
        tf.write(sc_value['df'].to_latex())

    # Bar plots comparing results
    sc_value['df'] = sc_value['df'].drop('conf', axis=1)
    df = sc_value['df'].set_index('method')
    df[df.columns.difference(list(scores.keys()))].plot(kind='bar')
    plt.title('Comparison of scores')
    plt.tight_layout()
    name = 'best_compare_{}'.format(sc_key)
    plt.savefig('results/{}/bar_{}.png'.format(datasets[id_data][:-5], name))
    plt.close()

    # for col in df:
    #     df[col].plot(kind='bar')
    #     plt.title(col)
    #     plt.tight_layout()
    #     name = 'best_{}_scoring_{}'.format(sc_key, col)
    #     plt.savefig('results/{}/bar_{}.png'.format(datasets[id_data][:-5], name))
    #     plt.close()

    # Radar plot comparing results
    df = sc_value['df']
    df = df[df.columns.difference(list(scores.keys()))]
    df = df.set_index('method').T.reset_index()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="polar")
    theta = np.arange(len(df) + 1) / float(len(df)) * 2 * np.pi
    for col in df:
        if col != 'index':
            values = df[col].values
            values = np.append(values, values[0])
            ax.plot(theta, values,
                    marker="o",
                    label=col)
    plt.xticks(theta[:-1], df['index'].apply(lambda x: x[:-6]))
    ax.tick_params(pad=10)
    ax.fill(theta, values, alpha=0)
    ax.set_ylim(0, math.ceil(values.max()*10)/10)

    plt.legend(bbox_to_anchor=(1, 0.5), loc="center right", fontsize=10,
               bbox_transform=plt.gcf().transFigure)
    plt.subplots_adjust(left=0.0, bottom=0.1, right=0.6)
    plt.title('Comparison of scores')
    plt.tight_layout()
    name = 'best_compare_{}'.format(sc_key)
    plt.savefig('results/{}/star_{}.png'.format(datasets[id_data][:-5], name))
    plt.close()
