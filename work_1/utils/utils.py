import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             adjusted_rand_score, fowlkes_mallows_score,
                             mutual_info_score, v_measure_score,
                             completeness_score, homogeneity_score)
from sklearn.neighbors import NearestNeighbors


def graph_epsilon(X, folder='other', name_file='other'):
    # Graph NN distances to aid choosing the DBSCAN eps parameter
    nn = NearestNeighbors(n_neighbors=2)
    nn = nn.fit(X)
    distances, indices = nn.kneighbors(X)
    distances = np.sort(distances, axis=0)[:, 1]
    plt.plot(distances)
    plt.title(name_file)
    plt.savefig(f'results/{folder}/epsilon_{name_file}.png')
    plt.close()


def compute_scores(X, true, pred):
    # Compute considered scores
    clusters = len(np.unique(pred))
    scores = {
        'silhouette_score':
            silhouette_score(X, pred) if clusters > 1 else None,
        'davies_bouldin_score':
            davies_bouldin_score(X, pred) if clusters > 1 else None,
        'adjusted_rand_score': adjusted_rand_score(true, pred),
        'fowlkes_mallows_score': fowlkes_mallows_score(true, pred),
        'homogeneity_score': homogeneity_score(true, pred),
        'mutual_info_score': mutual_info_score(true, pred),
        'v_measure_score': v_measure_score(true, pred),
        'completeness_score': completeness_score(true, pred),
     }
    return scores


# Change categorical columns to ordinal values equally
# Input: Dataframe with columns to be changed, categories in ascendant order, new values range
# Returns: Dataframe with ordinal columns
def ordinal_vote_representation(df, ls_categories, ls_range):
    ordinal_values = np.linspace(ls_range[0], ls_range[1], len(ls_categories))
    df.replace(ls_categories, ordinal_values, inplace=True)
    return df


# Visualize clusters over first two principal components
# Input: dataframe with n-1 columns with numerical values and 1 categorical class
def pca_visualization(df, cat_column, folder='other', name_file='other'):
    # Split X columns
    df_x = df.loc[:, df.columns != cat_column]

    # Do PCA and store first 2 components
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_x)
    exp_var = pca.explained_variance_ratio_
    pca_result = pd.DataFrame(pca_result)
    pca_result.columns = ['P1', 'P2']

    # Plot
    np.random.seed(4)
    for cluster in df[cat_column].unique():
        df_i = pca_result[df[cat_column] == cluster]
        plt.scatter(df_i['P1'], df_i['P2'], label=cluster, color=np.random.rand(3,))
    plt.legend(loc='center right', bbox_to_anchor=(1.15, 0.5))
    plt.xlabel(f'1st PCA component ({round(exp_var[0]*100, 2)}% explained variance)')
    plt.ylabel(f'2nd PCA component ({round(exp_var[1]*100, 2)}% explained variance)')
    plt.title(name_file)
    plt.savefig(f'results/{folder}/pca_{name_file}.png')
    plt.close()
