from numpy import linalg as LA
import numpy as np
import pdb


class PCA:

    def __init__(self, *, n_components=None, variance_cut=True):
        self.n_components = n_components
        self.variance_cut = variance_cut
        self.X_df = None
        self.mean_s = None
        self.t_X = None
        self.mean_s = None
        self.cov_df = None
        self.e_val = None
        self.e_vec = None
        self.selected_e_val = None
        self.selected_e_vec = None
        self.var_explained = None
        self.selected_var_explained = None

    def fit_transform(self, X):
        self.X_df = X.copy()

        self.mean_s = self.X_df.mean()  # Mean matrix
        self.X_df -= self.mean_s  # Center data for PCA
        self.cov_df = self.X_df.cov()  # Covariances matrix

        self.e_val, e_vec = LA.eig(self.cov_df)
        self.e_val = np.sort(self.e_val)[::-1]

        self.e_vec = e_vec.T[self.e_val.argsort()][::-1].T

        self.var_explained = self.e_val / self.e_val.sum()

        if not self.n_components and self.variance_cut:
            n = len(self.X_df.columns)
            cut = 1/n
            self.n_components = len(self.var_explained[self.var_explained > cut])

        if self.n_components:
            if self.n_components < 2:
                self.n_components = 2
            self.selected_e_val = self.e_val[:self.n_components]
            self.selected_e_vec = self.e_vec[:, :self.n_components]
            self.selected_var_explained = self.var_explained[:self.n_components]
        else:
            self.selected_e_val = self.e_val
            self.selected_e_vec = self.e_vec
            self.selected_var_explained = self.var_explained

        t_X = np.dot(self.X_df, self.selected_e_vec)

        self.X_df = self.X_df
        self.t_X = t_X
        self.mean_s = self.mean_s.to_numpy()
        return t_X

    def recover(self):
        rec_X_df = np.dot(self.X_df.T, self.t_X)
        rec_X_df = rec_X_df + self.mean_s[:min(rec_X_df.shape)]
        return rec_X_df
