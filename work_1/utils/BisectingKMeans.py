import numpy as np
import pandas as pd
from scipy import stats
from utils.KMeans import KMeans


class BisectingKMeans:

    def __init__(self, method='kmeans', seed=0):
        np.random.seed(seed)
        self.kmeans = KMeans(method, seed)

    def fit(self, X, n_clusters=8, n_init=10, max_iter=300):
        labels = np.zeros(len(X.index), dtype=int)
        for cuts in range(n_clusters-1):
            # choose cluster
            lrgst = stats.mode(labels)[0]
            # separate cluster data
            X_s = X[labels == lrgst]
            # apply kmeans
            new_labels, err = self.kmeans.fit(X_s, 2, n_init, max_iter)
            # update original data
            not_lrgst = np.delete(np.unique(labels), lrgst)
            i = 0
            aux = labels.copy()
            for v in not_lrgst:
                labels[aux == v] = i
                i += 1
            new_labels += cuts
            labels[aux == lrgst] = new_labels
        return labels
