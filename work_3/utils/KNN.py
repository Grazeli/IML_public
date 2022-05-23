import pandas as pd
import itertools
import numpy as np
from ReliefF import ReliefF
from sklearn.feature_selection import mutual_info_classif


# Distances one vs many for broadcasting
def minkowski_one_vs_many(np_i, np_train, mink_power, w=None):
    # Broadcasting, then summing each row, finally using the given power
    if w is None:
        w = 1.0
    dists = (w * abs(np_i - np_train) ** mink_power).sum(axis=1) ** (1 / mink_power)
    return dists


def cosine_one_vs_many(np_i, np_train, w=None, **kwargs):
    # Broadcasting product, adding by row
    if w is None:
        w = 1.0
    numerator = (w * np_i * np_train).sum(axis=1)
    t1 = np.sqrt(w * (np_i ** 2)).sum()
    t2 = np.sqrt(w * (np_train ** 2)).sum(axis=1)
    denominator = t1 * t2
    dists = (1 - numerator / denominator)
    return dists


def decide_IDW(l):
    l['id'] = 1/l['d']
    votes = l.groupby('y')['id'].sum()
    return votes.idxmax()  # takes first max in case of tie


def decide_sheppard(l):
    l['id'] = np.exp(-l['d'])
    votes = l.groupby('y')['id'].count()
    return votes.idxmax()  # takes first max in case of tie


def weight_relieff(X, y):
    fs = ReliefF(n_neighbors=100, n_features_to_keep=5)
    fs.fit(X.reset_index(drop=True).values,
           y.reset_index(drop=True))
    sc = fs.feature_scores
    n_sc = (sc - min(sc)) / (max(sc) - min(sc))
    return n_sc


def weight_IG(X, y):
    ig = mutual_info_classif(X, y)
    n_ig = (ig - min(ig)) / (max(ig) - min(ig))
    return n_ig


class kNN:

    metrics = {
        'minkowski': minkowski_one_vs_many,
        'cosine': cosine_one_vs_many,
    }
    decision_func = {
        'majority': lambda x: x[['y']].mode().values[0][0],
        'IDW': decide_IDW,
        'sheppard': decide_sheppard,
    }
    weight_func = {
        'equal': lambda x, y: None,
        'relieff': weight_relieff,
        'IG': weight_IG,
    }
    powers = [1, 2]

    def __init__(self, k, distance, decision, weighting, mink_power=None):
        if distance not in self.metrics.keys():
            raise ValueError
        if decision not in self.decision_func.keys():
            raise ValueError
        if weighting not in self.weight_func.keys():
            raise ValueError
        if mink_power and mink_power not in self.powers:
            raise ValueError
        self.distance = self.metrics[distance]
        self.decision = self.decision_func[decision]
        self.weight = self.weight_func[weighting]
        self.mink_power = mink_power
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.w = self.weight(self.X_train, self.y_train)

    def get_pred(self, sample):
        dists = self.distance(
            **{'np_i': sample.values,
               'np_train': self.X_train.values,
               'w': self.w,
               'mink_power': self.mink_power})
        labeled = pd.DataFrame({'d': dists, 'y': self.y_train})
        labeled = labeled.sort_values('d')
        labeled = labeled[:self.k]
        res = self.decision(labeled)
        return res

    def predict(self, X):
        preds = X.apply(self.get_pred, axis=1)
        return preds

