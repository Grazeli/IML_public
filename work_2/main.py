import numpy as np
import pandas as pd
from utils.utils import *
from utils.KMeans import *
from data_preprocessing import *
import itertools
import matplotlib.pyplot as plt

from utils.PCA import *
from sklearn.decomposition import PCA as skPCA
from sklearn.decomposition import IncrementalPCA as skIncPCA
from sklearn.manifold import TSNE

# from pca import pca as pcaLib

# Parameters of execution
id_data = 1  # Select the desired data set with a value of 0, 1 or 2
datasets = ['vote', 'breast-w', 'heart-c']
y_var = ['Class', 'Class', 'num']

# Read and preprocess data
path_datasets = 'datasets/'
dataset_path = path_datasets + datasets[id_data] + '.arff'
df = load_arff_data(dataset_path)
df_X, df_y = preprocessing(df, datasets[id_data] + '.arff', y_var[id_data])
df = df_X.copy()

# Implemented PCA
pca = PCA(variance_cut=True)
princomp = pca.fit_transform(df_X)

# Use the number of components for sklearn PCAs
n_comp = princomp.shape[1]
print('Number of features: {}'.format(len(df_X.columns)))
print('Number of components: {}'.format(n_comp))

skpca = skPCA(n_components=n_comp)
skprincomp = skpca.fit_transform(df_X)

skincpca = skIncPCA(n_components=n_comp)
skincprincomp = skincpca.fit_transform(df_X)

# # UNNCOMENT TO EXECUTE LOADINGS IF LIBRARY INSTALLED
# libpca = pcaLib(n_components=n_comp)  # Used to plot loadings for latent variables interpretation
# reslibpca = libpca.fit_transform(df_X)
# # Plot loadings
# fig, ax = libpca.biplot(cmap=None, n_feat=16, legend=False, label=False, d3=False, figsize=(8, 6))
# fig.savefig('results/{}/pca_loadings.png'.format(datasets[id_data]))

# Print requested PCA steps in console
print('\nCovariance matrix: \n', pca.cov_df)  # printed as requested
print(f'\n{len(pca.e_val)} Eigen values in descending order:\n', pca.e_val)
print('\nVariance proportion explained by the principal components in descending order:\n', pca.var_explained)
df_e_vec = pd.DataFrame(pca.e_vec)
print(f'\n{len(pca.e_val)} Eigen vectors from the covariance matrix sorted in order according to the eigen values: \n',
      df_e_vec)

# Analyze graph of components represented variance
fig, ax = plt.subplots()
component_numbers = list(range(1, len(pca.e_val) + 1))
ax.plot(component_numbers, pca.var_explained * 100, label='Variance')
ax.plot(component_numbers, np.cumsum(pca.var_explained) * 100, label='Cumulative Variance')
ax.set(xlabel='Principal Component', ylabel='Explained Variance (%)',
       title='Principal Components explained variance')
ax.grid()
ax.legend()
fig.savefig('results/{}/pca_variance.png'.format(datasets[id_data]))

# Graph also sorted eigenvalues
fig, ax = plt.subplots()
component_numbers = list(range(1, len(pca.e_val) + 1))
ax.plot(component_numbers, pca.e_val, label='Eigenvalues')
ax.set(xlabel='Principal Component', ylabel='Eigenvalues',
       title='Principal Components eigenvalues')
ax.grid()
ax.legend()
fig.savefig('results/{}/pca_eigenval.png'.format(datasets[id_data]))

# Print PCA dimensionality reduction results
print(f'\n{len(pca.selected_e_val)} Selected Eigen values in descending order:\n', pca.selected_e_val)
print('\nVariance proportion explained by the principal components in descending order:\n', pca.selected_var_explained)
print('\nTotal Variance explained by selected components:', pca.selected_var_explained.sum())
df_selected_e_vec = pd.DataFrame(pca.selected_e_vec)
print(f'\n{len(pca.selected_e_val)} Selected Eigen vectors from the covariance matrix sorted in order according to the eigen values: \n',
      df_selected_e_vec)

# Change direction of components to ease the comparison
for comp in [princomp, skincprincomp]:
    for i in range(comp.shape[1]):
        if (comp[:, i] * skprincomp[:, i] < 0).any():
            comp[:, i] *= -1

# Plot all PCA results to compare the methods
fig, axs = plt.subplots(1, 3, figsize=(10, 6))
axs[0].scatter(princomp[:, 0], princomp[:, 1])
axs[0].set(title='Implemented PCA')
axs[1].scatter(skprincomp[:, 0], skprincomp[:, 1])
axs[1].set(title='sklearn PCA')
axs[2].scatter(skincprincomp[:, 0], skincprincomp[:, 1])
axs[2].set(title='sklearn IncrementalPCA')
fig.savefig('results/{}/comparison_pca_methods.png'.format(datasets[id_data]))

# Plot recovered data
fig, axs = plt.subplots(1, 2)
recovered = pca.recover()
axs[0].scatter(df_X.iloc[:, 0], df_X.iloc[:, 1])
axs[0].set(title='Original data')
axs[1].scatter(recovered[:, 0], recovered[:, 1])
axs[1].set(title='Recovered data')
fig.savefig('results/{}/comparison_recovered.png'.format(datasets[id_data]))

# Number of cluster configurations used
n_clusters_conf = np.arange(2, 7, 1)
# DF to store results
results = pd.DataFrame()
# Datasets, to store labels and plot them later
datas = {
    'original data and KMeans labels':
        {
            'X': df_X,
            'Y': None
        },
    'PCA reduced data and KMeans labels':
        {
            'X': pd.DataFrame(princomp),
            'Y': None
        }
}
for key, ds in datas.items():
    data = ds['X']
    for i, n_clus in enumerate(n_clusters_conf):
        # Initialize model
        kmeans = KMeans('kmeans')
        # Fit model
        pred_labels, _ = kmeans.fit(data, n_clus)

        # Score results and add them to the DataFrame
        scores = compute_scores(data, df_y.T.values[0], pred_labels)
        results = results.append({'name': '{}_{}_{}'.format('KMeans', key, i),
                                  'dataset': key, 'labels': pred_labels,
                                  'n_clusters': n_clus,
                                  **scores},
                                 ignore_index=True)

    # Save best labels selected by Silhouette
    current_res = results[results['dataset'] == key]
    best_sil_row = current_res['silhouette_score'].argmax()
    ds['Y'] = current_res.iloc[best_sil_row]['labels']

# Save results, in both csv and tex
file = 'results/{}/{}'.format(datasets[id_data], datasets[id_data])

results.to_csv('{}.csv'.format(file))
with open('{}.tex'.format(file), 'w') as tf:
    tf.write(results.T.to_latex())

# Add original data and true labels to datasets
datas['original data'] = {'X': df_X,
                          'Y': df_y[y_var[id_data]].cat.codes.to_numpy()}

# Plot PCA and TSNE for the 3 datasets obtained
# Parameters tunned
to_tune = {
    'early_exaggeration': [2.0, 20.0, 50.0],
    'learning_rate': [10.0, 100.0, 300.0],
}
# Perplexities considered
perplexities = [5, 10, 20, 40]
# Repetitions due to randomness
repetitions = 10

# Generate all combinations of parameter values
keys, values = zip(*to_tune.items())
tune_perm = [dict(zip(keys, v)) for v in itertools.product(*values)]

# The following allows not to perform the TSNE calculations on
# original data two times just to change labels
save_original = [None] * len(perplexities)
for key, ds in datas.items():
    X = ds['X']
    Y = pd.Series(ds['Y'])

    c = ['r', 'g', 'b', 'c', 'k', 'w']
    Y_colors = Y.apply(lambda x: c[x])

    # TSNE fist
    fig, axs = plt.subplots(2, int(len(perplexities)/2), sharex=True, sharey=True, figsize=(6, 6))
    fig.suptitle('T_SNE visualization for different\n perplexities using {}.'.format(key))

    row_index = 0
    col_index = 0

    # Repeat for all perplexities as it is not a tunnable parameter
    for i, per in enumerate(perplexities):
        print(f'perplexity {per}')

        selected_kl = None
        selected_df = None
        if 'original' in key and save_original[i] is None:
            # Check all combinations and save the best results
            for params in tune_perm:
                print(f'\tparams {params}')
                tsne = TSNE(perplexity=per, **params)

                # Repetitions due to randomness
                for rep in range(repetitions):
                    print(f'\t\trep {rep}')
                    df_tsne = tsne.fit_transform(X)
                    kl_div = tsne.kl_divergence_
                    print(f'\t\tobtained kl {kl_div}')
                    if selected_kl is None or kl_div < selected_kl:
                        selected_kl = kl_div
                        selected_df = df_tsne
            save_original[i] = selected_df
        else:
            selected_df = save_original[i]

        axs[row_index, col_index].scatter(x=selected_df[:, 0],
                                          y=selected_df[:, 1], c=Y_colors)
        axs[row_index, col_index].set_title('Perplexity {}'.format(per))

        col_index += 1
        if col_index >= len(perplexities)/2:
            col_index = 0
            row_index += 1
    fig.savefig('results/{}/tsne_{}.png'.format(datasets[id_data], key))

    # PCA plot
    pca = PCA(n_components=n_comp)
    pca_proj = pca.fit_transform(df_X)

    fig, axs = plt.subplots(1, 1, figsize=(10, 6))
    axs.scatter(pca_proj[:, 0], pca_proj[:, 1], c=Y_colors)
    axs.set(title='PCA visualization using {}.'.format(key))
    fig.savefig('results/{}/pca_{}.png'.format(datasets[id_data], key))
