import numpy as np


def reindex_monotonic(X_mis):
    return np.argsort(np.sum(~np.isnan(X_mis), axis=0))[::-1]


def all_mdps(X_mis):
    mdps = np.unique(np.isnan(X_mis), axis=0)
    mdps = mdps[np.any(mdps, axis=1), :]  # remove complete cases
    mdps = mdps[np.argsort(np.count_nonzero(mdps, axis=1)), :]  # sort by increasing missingness
    return mdps


def mdp_coords(X_mis, mdp):
    return np.where((np.isnan(X_mis) == mdp).all(axis=1))[0]


def old_to_new(old_indices, new_indices):
    return {i: j for i, j in zip(old_indices, new_indices)}


def missing_rows(X_mis):
    return np.any(np.isnan(X_mis), axis=1)


def n_miss_by_col(X_mis):
    return np.sum(np.isnan(X_mis), axis=0)
