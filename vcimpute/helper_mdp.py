import numpy as np


def reindex_monotonic(X_mis):
    return np.argsort(np.sum(~np.isnan(X_mis), axis=0))[::-1]


def all_miss_vars(X_mis):
    a = np.zeros(np.prod(X_mis.shape), dtype=np.uint64)
    b = np.flatnonzero(np.isnan(X_mis))
    a[b] = 1 + b % X_mis.shape[1]
    a = a.reshape(X_mis.shape)
    a = np.unique(a, axis=0)
    a = a[~np.all(a == 0, axis=1)]  # remove complete cases
    return a


def all_mdps(X_mis):
    mdps = np.unique(np.isnan(X_mis), axis=0)
    mdps = mdps[np.any(mdps, axis=1), :]  # remove complete cases
    return mdps


def miss_vars_to_mdp(miss_vars, d):
    mdp = np.zeros(shape=(d,), dtype='bool')
    mdp[np.array(miss_vars) - 1] = True
    return mdp


def sort_mdps_by_increasing_missing_vars(mdps):
    return mdps[np.argsort(np.count_nonzero(mdps, axis=1)), :]


def sort_miss_vars_by_increasing_miss_vars(miss_vars):
    n_miss_vars = list(map(len, miss_vars))
    miss_vars = np.array(miss_vars, dtype='object')
    return list(miss_vars[np.argsort(n_miss_vars)])


def mdp_coords(X_mis, mdp):
    return np.where((np.isnan(X_mis) == mdp).all(axis=1))[0]


def old_to_new(old_indices, new_indices):
    return {i: j for i, j in zip(old_indices, new_indices)}


def count_missing_by_row(X_mis):
    return np.sum(np.isnan(X_mis), axis=1)


def count_missing_by_col(X_mis):
    return np.sum(np.isnan(X_mis), axis=0)
