import pandas as pd
import numpy as np
from scipy.spatial import distance


def computeCentroids(df, y_name, notVarCols):
    labels = df[y_name].unique()
    centroids = pd.DataFrame()

    for l in labels:
        center = df[df[y_name] == l].drop(notVarCols, axis=1).mean()
        df_label = df[df[y_name] == l]

        distancesToCenter = distance.cdist(df_label.drop(notVarCols, axis=1), [center])  # By default euclidean
        indexToCloser = distancesToCenter.argmin()
        closest = df_label.iloc[indexToCloser]

        centroids = centroids.append(closest)

    return centroids


class FCNN:
    # notVarCols represents the columns that should not be used for the distance between two points.
    def __init__(self, notVarCols=[]):
        self.notVarCols = notVarCols
        pass

    def transform(self, T, y_name, k):
        S = pd.DataFrame()
        auxS = computeCentroids(T, y_name, self.notVarCols)

        # Create nearest matrix: k nearest neighbors for each element of T ordered.
        nearest = [[None] * k] * T.shape[0]

        print('Start While loop:')
        while not auxS.empty:
            print('---- Start while iteration ----')
            print('Original data shape: ', T.shape)
            print('S shape', S.shape)

            # Compute distance S x auxS

            # Keys are tuple (index in S, index in auxS)
            distanceS = {}

            S = pd.concat([S, auxS])
            # Remove from T the elements of auxS (As they are in S now)
            T.drop(auxS.index, inplace=True)

            for idxp, p in S.iterrows():
                for idxa, a in auxS.iterrows():
                    distanceS[(idxp, idxa)] = distance.euclidean(p.drop(self.notVarCols), a.drop(self.notVarCols))

            # Initialize to undefined the Rep dictionary assigning to the index of s an index of T
            rep = {s: None for s in S.index}
            voren = {s: [] for s in S.index}

            # Update nearest matrix for each point in T
            print('Iteration on Training data')
            for idxq, q in T.iterrows():
                current_nearest = nearest[idxq].copy()
                for idxp, p in auxS.iterrows():
                    index = k - 1
                    inserted = False
                    if current_nearest[-1] is None or distance.euclidean(
                            S.loc[current_nearest[-1]].drop(self.notVarCols), q.drop(self.notVarCols)) * 2 > distanceS[
                        (current_nearest[-1], idxp)]:
                        distQP = distance.euclidean(q.drop(self.notVarCols), p.drop(self.notVarCols))
                        while index >= 0 and not inserted:
                            if current_nearest[index] != None and distance.euclidean(
                                    S.loc[current_nearest[index]].drop(self.notVarCols),
                                    q.drop(self.notVarCols)) < distQP:
                                # Need to be inserted to position index+1
                                current_nearest.insert(index + 1, idxp)
                                inserted = True
                            index -= 1
                        # If it has not been inserted, need to be inserted at the first position
                        if not inserted:
                            current_nearest.insert(0, idxp)

                nearest[idxq] = current_nearest[:k]

                # In for each q: see if can be a rep
                label_nearest = []
                for idx in nearest[idxq]:
                    if not idx is None:
                        label_nearest.append(S.loc[idx][y_name])

                label_majority = max(label_nearest, key=label_nearest.count)
                if q[y_name] != label_majority:
                    voren[nearest[idxq][0]].append(idxq)

            for key in voren:
                if len(voren[key]) != 0:
                    temp_df = T.loc[voren[key]]
                    center = temp_df.drop(self.notVarCols, axis=1).mean()
                    distancesToCenter = distance.cdist(temp_df.drop(self.notVarCols, axis=1), [center])

                    indexToCloser = distancesToCenter.argmin()
                    closest = temp_df.iloc[indexToCloser]

                    rep[key] = closest.name

            print('Extract future auxS')
            auxS = pd.DataFrame()
            for key in rep:
                if not rep[key] is None:
                    candidate = T.loc[[rep[key]]]
                    auxS = pd.concat([auxS, candidate])

            print('---- End iteration loop ----')

            print(S)
        return S