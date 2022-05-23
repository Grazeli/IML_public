import pandas as pd
# pd.options.mode.chained_assignment = None
import numpy as np
import itertools
import random
import pdb

from utils.KNN import kNN


class MENN(kNN):
    def __init__(self, k_list=None):
        if k_list is None:
            k_list = [1, 3, 5]
        self.distance = self.metrics['minkowski']
        self.decision = self.decision_func['majority']
        self.weight = self.weight_func['equal']
        self.mink_power = 2
        self.k = k_list

    def fit_transform(self, X, y):
        a = self.k
        b = self.k
        if a[0] == 1:
            a = a[1:]
        k_vals = list(itertools.product(a, b))
        min_misc = None
        reduced_X = None
        reduced_y = None
        for k1, k2 in k_vals:
            print(k1, k2)
            self.k = k1
            self.X_train = X.copy()  # W0
            self.y_train = y.copy()
            typical = X.apply(self.get_typical, axis=1)

            # edited reference sets
            self.X_train = self.X_train[typical]  # W1(k)
            self.y_train = self.y_train[typical]

            self.k = k2
            y_pred = X.apply(self.get_pred, axis=1)
            misc = (y_pred != y).sum()
            if not min_misc or misc < min_misc:
                min_misc = misc
                reduced_X = self.X_train.copy()
                reduced_y = self.y_train.copy()
        return reduced_X, reduced_y

    def get_pred(self, sample):
        dists = self.distance(
            **{'np_i': sample.values,
               'np_train': self.X_train.values,
               'mink_power': self.mink_power})
        labeled = pd.DataFrame({'d': dists, 'y': self.y_train})
        labeled = labeled.sort_values('d')
        nearest = labeled.iloc[0]['d']
        idx = int(nearest == 0)
        labeled = labeled[idx:self.k+idx]
        res = self.decision(labeled)
        return res

    def get_typical(self, sample):
        dists = self.distance(
            **{'np_i': sample.values,
               'np_train': self.X_train.values,
               'mink_power': self.mink_power})
        labeled = pd.DataFrame({'d': dists, 'y': self.y_train})
        labeled = labeled.sort_values('d')
        l = labeled.iloc[self.k-1]['d']
        labeled = labeled[labeled['d'] <= l]
        return (labeled['y'] == labeled['y'].iloc[0]).all()


class IB3(kNN):
    def __init__(self, thres_acceptance=0.9, thres_drop=0.7):
        self.distance = self.metrics['minkowski']
        self.decision = self.decision_func['majority']
        self.weight = self.weight_func['equal']
        self.mink_power = 2
        self.thres_acc = thres_acceptance
        self.thres_drop = thres_drop
        self.S_meta = pd.DataFrame(columns=['id', 'n_attempts', 'n_acc', 'p_acc',
                                            'u_bound', 'l_bound',
                                            'u_bound_class', 'l_bound_class']) #For drop or acceptance criteria
        self.S_X = None
        self.S_y = None

    def fit_transform(self, X, y):
        X.reset_index(inplace=True)
        y.reset_index(inplace=True, drop=True)
        self.S_X = pd.DataFrame(columns=X.columns) # Start with empty set
        self.S_y = pd.Series(name=y.name)

        # Iterate over rows, as it is a sequential algorithm adding one sample at a time, not possible to paralelize
        for i in range(X.shape[0]):
            print(f'i: {i}/{X.shape[0]}')
            if i == 0: # Start by adding first sample
                self.S_X = self.S_X.append(X.iloc[[i], :])
                self.S_y = self.S_y.append(y.iloc[[i]])
                self.S_meta = self.S_meta.append({'id': i, 'n_attempts': 0, 'n_acc': 0, 'p_acc': 0,
                                                  'u_bound': None, 'l_bound': None,
                                                  'u_bound_class': None, 'l_bound_class': None}, ignore_index=True)

            else:
                # t is instance i from X and y
                # Estimate distances from previous S instances to t
                dist_t_S = self.get_pred(sample=X.iloc[i, :])
                # Select acceptable cases
                if i == 1: # There is still not previous record to be used
                    bool_acc = [False]
                else:
                    bool_acc = self.S_meta['l_bound'] > self.S_meta['u_bound_class'] # TODO is correct
                df_acc = self.S_meta[bool_acc] # Acceptable cases

                # Get nearest acceptable instance
                if df_acc.shape[0] == 0:  # If there are not acceptable instances in S, use a random instance in S
                    id_acc = random.randint(0, self.S_meta.shape[0] - 1)
                    # id_acc = self.S_meta['id'][id_acc]
                    df_acc = self.S_meta.iloc[[id_acc], :]
                    y_chosen = y[[df_acc['id'].iloc[0]]]
                    dist_t_a = dist_t_S.iloc[id_acc, :]['d']
                    print('random')
                else:  # Closer, min distance
                    id_acc = dist_t_S[(dist_t_S['d'] == dist_t_S['d'].min()).values].index[0]  # unique row
                    y_chosen = y[[id_acc]]
                    dist_t_a = dist_t_S.loc[id_acc, :]['d']

                # For each instance in S, if s at least as close to t as a, update its record
                cases_update = dist_t_S['d'] <= dist_t_a  # Meet distance criteria
                self.S_meta['n_attempts'][cases_update.values] = self.S_meta['n_attempts'][cases_update.values] + 1  # Update number of attempts
                cases_update2 = dist_t_S['y'] == y[i]  # Meet category criteria
                self.S_meta[cases_update.values & cases_update2.values]['n_acc'] = \
                    self.S_meta[cases_update.values & cases_update2.values]['n_acc'] + 1  # Update good cases
                self.S_meta['p_acc'] = self.S_meta['n_acc'] / self.S_meta['n_attempts']  # Update accuracy
                self.S_meta['l_bound'] = self.S_meta.apply(
                    lambda x: conf_int(x['p_acc'], x['n_attempts'], self.thres_acc, 'lower'), axis=1)
                self.S_meta['u_bound'] = self.S_meta.apply(
                    lambda x: conf_int(x['p_acc'], x['n_attempts'], self.thres_drop, 'upper'), axis=1)

                # Add data from category for comparisons
                l_prev_columns = ['id', 'n_attempts', 'n_acc', 'p_acc', 'u_bound', 'l_bound']
                l_prev_columns_class = [y.name, 'u_bound_class', 'l_bound_class']
                df_S_metay = pd.concat([self.S_meta[l_prev_columns].reset_index(drop=True), self.S_y.reset_index(drop=True)], axis=1)# dataframe adding the class
                #Summarize for each class
                df_sum_cat = df_S_metay.groupby(y.name, as_index=False).agg({'n_attempts': 'count', 'n_acc': 'count'})
                df_sum_cat['p_acc'] = df_sum_cat['n_acc'] / df_sum_cat['n_attempts']  # Update accuracy
                df_sum_cat['u_bound_class'] = df_sum_cat.apply(
                    lambda x: conf_int(x['p_acc'], x['n_attempts'], self.thres_acc, 'upper'), axis=1)
                df_sum_cat['l_bound_class'] = df_sum_cat.apply(
                    lambda x: conf_int(x['p_acc'], x['n_attempts'], self.thres_drop, 'lower'), axis=1)


                # Compare each s bound with its category bounds to frop those meeting the criteria
                # The acceptables are chosen in line 116 starting the loop
                df_S_metay = pd.merge(df_S_metay, df_sum_cat[l_prev_columns_class], how='left', on=y.name)  # Merge class bounds for comparison, by class
                cases_keep = df_S_metay['u_bound'] > df_S_metay['l_bound_class']  # TODO check
                # Update, dropping those not meeting
                df_S_metay = df_S_metay[cases_keep.values]
                self.S_X = self.S_X[cases_keep.values]
                self.S_y = self.S_y[cases_keep.values]

                self.S_meta = df_S_metay[df_S_metay.columns.difference([y.name])]  #Remove y again
                # Compare class of 'a' with 't'.
                class_t = y[i]
                class_a = y_chosen.iloc[0]
                if class_t != class_a:  # Add t to S
                    self.S_X = self.S_X.append(X.iloc[[i], :])
                    self.S_y = self.S_y.append(y.iloc[[i]])
                    self.S_meta = self.S_meta.append({'id': i, 'n_attempts': 1, 'n_acc': 1, 'p_acc': 1,
                                                      'u_bound': 1, 'l_bound': 0.671,
                                                      'u_bound_class': 0, 'l_bound_class': 1}, ignore_index=True)
            print(self.S_meta)
        return self.S_X, self.S_y

    def get_pred(self, sample):
        dists = self.distance(
            **{'np_i': sample.values,
               'np_train': self.S_X.values,
               'mink_power': self.mink_power})
        labeled = pd.DataFrame({'d': dists, 'y': self.S_y})
        return labeled


def conf_int(p, n, z, bound):
    if bound == 'lower':
        if n == 0:
            res = 1 # not started yet
        else:
            term1 = p + z ** 2 / (2 * n)
            term2 = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
            term3 = 1 + z ** 2 / n
            res = (term1 - term2) / term3
    else:
        if n == 0:
            res = 0 # not started yet
        else:
            term1 = p + z ** 2 / (2 * n)
            term2 = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
            term3 = 1 + z ** 2 / n
            res = (term1 + term2) / term3
    return res


class IB2(kNN):
    def __init__(self, thres_acceptance=0.9, thres_drop=0.7):
        self.distance = self.metrics['minkowski']
        self.decision = self.decision_func['majority']
        self.weight = self.weight_func['equal']
        self.mink_power = 2
        self.S_X = None
        self.S_y = None

    def fit_transform(self, X, y):
        X.reset_index(inplace=True)
        y.reset_index(inplace=True, drop=True)
        self.S_X = pd.DataFrame(columns=X.columns) # Start with empty set
        self.S_y = pd.Series(name=y.name)

        # Iterate over rows, as it is a sequential algorithm adding one sample at a time, not possible to paralelize
        for i in range(X.shape[0]):
            # print(f'i: {i}/{X.shape[0]}')
            if i == 0: # Start by adding first sample
                self.S_X = self.S_X.append(X.iloc[[i], :])
                self.S_y = self.S_y.append(y.iloc[[i]])
                continue

            else:
                # t is instance i from X and y
                # Estimate distances from previous S instances to t
                dist_t_S = self.get_pred(sample=X.iloc[i, :])
                # Get nearest instance
                id_acc = dist_t_S[(dist_t_S['d'] == dist_t_S['d'].min()).values].index[0]  # unique row
                y_chosen = y[[id_acc]]
                dist_t_a = dist_t_S.loc[id_acc, :]['d']

                class_t = y[i]
                class_a = y_chosen.iloc[0]
                if class_t != class_a:  # Add t to S
                    self.S_X = self.S_X.append(X.iloc[[i], :])
                    self.S_y = self.S_y.append(y.iloc[[i]])

        self.S_X.drop(columns=['index'], inplace=True)
        return self.S_X, self.S_y

    def get_pred(self, sample):
        dists = self.distance(
            **{'np_i': sample.values,
               'np_train': self.S_X.values,
               'mink_power': self.mink_power})
        labeled = pd.DataFrame({'d': dists, 'y': self.S_y})
        return labeled