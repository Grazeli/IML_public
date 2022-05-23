import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances


class KMeans:

    def __init__(self, method='kmeans', seed=0):
        # Different configuration for each method
        self.conf = {
            'kmeans': {
                'cent_init': self._centroids_init,
                'step': self._step,
                'step_op': 'mean'
            },
            'kmeans++': {
                'cent_init': self._centroids_init_plus,
                'step': self._step,
                'step_op': 'mean'
            },
            'kmedians': {
                'cent_init': self._centroids_init,
                'step': self._step,
                'step_op': 'median'
            },
        }

        np.random.seed(seed)

        # Save the configuration initialised
        if method not in self.conf.keys():
            raise ValueError('Method not found')
        self.cent_init = self.conf[method]['cent_init']
        self.step = self.conf[method]['step']
        self.step_op = self.conf[method]['step_op']

    def _centroids_init(self, X, n):
        # Initialise random centroids
        c = np.random.uniform(low=round(X.values.min(), 0),
                              high=round(X.values.max(), 0),
                              size=(n, len(X.columns)))
        return pd.DataFrame(c)

    def _centroids_init_plus(self, X, n):
        centroids = X.iloc[[np.random.randint(X.shape[0])], :]  # random first point
        for centroid in range(n - 1):
            distances_matrix = euclidean_distances(X, centroids)
            min_dist_id = np.argmin(distances_matrix, axis=1)  # For each point the closest cluster
            # Select the maximum distance from all points to its closest cluster
            max_closest_id = np.argmax(distances_matrix[np.arange(len(distances_matrix)), min_dist_id], axis=0)
            max_dist_point = X.iloc[[max_closest_id], :]
            centroids = centroids.append(max_dist_point)
        centroids.columns = list(range(centroids.shape[1]))
        centroids.reset_index(drop=True, inplace=True)
        return centroids

    def _step(self, X, cent):
        # assign point to closest centroid
        lbls = np.argmin(euclidean_distances(X, cent), axis=1)
        # calculate new centroids
        aux = cent.copy()
        cent = cent.apply(
            lambda x: getattr(X[lbls == x.name],
                              self.step_op)(),
            axis=1)
        nan_idx = cent[cent.iloc[:, 0].isnull()].index.tolist()
        cent.iloc[nan_idx] = aux.iloc[nan_idx]
        error = cent.apply(lambda x: np.sum(
            (X[lbls == x.name] - x.iloc[0])**2), axis=1).values.sum()
        cent = pd.DataFrame(cent)
        return lbls, cent, error

    def fit(self, X, n_clusters=8, n_init=10, max_iter=300):
        best_error = None
        best_labels = None
        prev_labels = None

        # Iterate repetitions with different initial centroids
        for i in range(n_init):
            centroids = self.cent_init(X, n_clusters)

            # Iterate maximum number of repetitions
            for j in range(max_iter):
                labels, centroids, error = self.step(X, centroids)

                # Stop if no changes
                if (labels == prev_labels).all():
                    break
                prev_labels = labels.copy()

            # Update best results found
            if best_labels is None:
                best_labels = labels
            if len(np.unique(labels)) == n_clusters and (
                    not best_error or error < best_error):
                best_error = error
                best_labels = labels

        return best_labels, best_error
