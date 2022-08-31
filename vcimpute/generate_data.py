import random

import numpy as np
import pyvinecopulib as pv
from gcimpute.helper_data import generate_sigma, generate_LRGC, generate_mixed_from_gc
from gcimpute.helper_mask import mask_MCAR as gcimpute_mask_MCAR
from statsmodels.distributions.empirical_distribution import ECDF

from vcimpute.util import make_triangular_array


def make_complete_data_matrix(n, d, copula_type, **kwargs):
    if copula_type == 'LRGC':
        assert 'sigma' in kwargs, 'LRGC needs param sigma'
        assert 'rank' in kwargs, 'LRGC needs param rank'
        X, _ = generate_LRGC(
            var_types={'cont': list(range(d))},
            rank=kwargs['rank'],
            sigma=kwargs['sigma'],
            n=n
        )
        U = obs_to_uniform(X)
    elif copula_type == 'gaussian' and ('vine_structure' not in kwargs):
        sigma = generate_sigma(p=d)
        X = generate_mixed_from_gc(
            var_types={'cont': list(range(d))},
            n=n,
            sigma=sigma
        )
        U = obs_to_uniform(X)
    elif copula_type in ('gaussian', 'student', 'clayton', 'frank'):
        assert 'vine_structure' in kwargs, 'copula sim needs param vine_structure'
        assert kwargs['vine_structure'] in ['C', 'D', 'R'], 'vine structure must be C, D or R'

        structure = None
        if kwargs['vine_structure'] == 'R':
            structure = pv.RVineStructure.simulate(d)
        elif kwargs['vine_structure'] == 'C':
            structure = pv.CVineStructure.simulate(d)
        elif kwargs['vine_structure'] == 'D':
            structure = pv.DVineStructure.simulate(d)

        family = None
        if copula_type == 'gaussian':
            family = pv.BicopFamily.gaussian
        elif copula_type == 'student':
            family = pv.BicopFamily.student
        elif copula_type == 'clayton':
            family = pv.BicopFamily.clayton
        elif copula_type == 'frank':
            family = pv.BicopFamily.frank

        pair_copulas = make_triangular_array(d)
        theta = generate_theta(d * (d - 1) // 2, copula_type)
        k = 0
        for j in range(d - 1):
            for i in range(d - j - 1):
                parameters = [[theta[k]], [2]] if copula_type == 'student' else [theta[k]]
                pair_copulas[i][j] = pv.Bicop(family=family, parameters=parameters)
                k += 1
        assert len(theta) == k

        cop = pv.Vinecop(structure, pair_copulas)
        U = cop.simulate(n)
    else:
        raise NotImplementedError('copula_type must be one of LRGC, gaussian, student, clayton, frank')
    return U


def mask_MCAR(X, mask_fraction, d_mis=None, monotonic_missingness=False):
    n = X.shape[0]
    d = X.shape[1]

    if d_mis == 1:
        X_mask = np.copy(X)
        miss_idx = random.choice(list(range(d)))
        is_missing = np.random.binomial(n=1, p=mask_fraction, size=X_mask.shape[0]).astype(bool)
        X_mask[is_missing, miss_idx] = np.nan
    elif monotonic_missingness:
        assert d_mis < d
        X_mask = np.copy(X)
        for j in range(d - d_mis, d):
            is_missing = np.random.binomial(n=1, p=mask_fraction, size=n).astype(bool)
            X_mask[is_missing, j:] = np.nan
    else:
        X_mask = gcimpute_mask_MCAR(X, mask_fraction=mask_fraction)
    return X_mask


def generate_theta(d, copula_type):
    assert copula_type in ['gaussian', 'student', 'clayton', 'frank']
    theta = None
    if copula_type in ['gaussian', 'student']:
        theta = 2 * (np.random.uniform(size=d) - 0.5)
    elif copula_type == 'frank':
        sign = 2 * (np.random.binomial(n=1, p=0.5, size=d) - 0.5)
        theta = sign * np.random.uniform(0, 35, size=d)
    elif copula_type == 'clayton':
        theta = np.random.uniform(1e-10, 28, size=d)
    return theta


def obs_to_uniform(X):
    U = np.empty(shape=X.shape)
    for j in range(X.shape[1]):
        U[:, j] = ECDF(X[:, j])(X[:, j])
    return U
