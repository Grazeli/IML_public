import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from numpy import linalg as LA


class FuzzyCMeans:

    def __init__(self, m=2, seed=0):
        self.m = m
        np.random.seed(seed)

    def fit(self, X, n_clusters=2, threshold=1e-5, max_iter=100, crisp=False):
        # Initialize centroids and compute U(0)
        centroids = self.initCentroids(X, n_clusters)
        previousU = np.zeros((X.shape[0], n_clusters))

        for t in range(max_iter):
            t += 1

            # Compute U with new centroids
            u = self.computeU(X, centroids)

            #Update centroids with new U
            centroids = self.updateCentroids(X, u)

            if LA.norm(u - previousU) < threshold:
                if crisp:
                    return centroids, self.toCrispClustering(u)
                else:
                    return centroids, u

            previousU = u

        if crisp:
            return centroids, self.toCrispClustering(u)
        else:
            return centroids, u

    def initCentroids(self, X, n):
        c = np.random.uniform(
            low=round(X.min()[0], 0),
            high=round(X.max()[0], 0),
            size=(n, len(X.columns))
        )
        return pd.DataFrame(c)

    def updateCentroids(self, X, u):
        upowerm = u ** self.m
        num = X.T @ upowerm
        denominator = np.sum(upowerm, axis=0)
        return (num / denominator).T

    def computeU(self, X, centroids):
        power = float(2 / (self.m - 1))
        U = np.zeros((X.shape[0], len(centroids)))
        i = 0
        for row in X.iterrows():
            x = np.array(row[1])
            temp = euclidean_distances([x], centroids)[0] ** power
            denominator = np.sum(np.reciprocal(temp))
            U[i] = np.reciprocal(temp * denominator)
            i += 1
        return U

    def toCrispClustering(self, U):
        partition = np.argmax(U, axis=1)
        return partition
