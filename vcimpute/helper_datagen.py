import numpy as np
import pyvinecopulib as pv
from gcimpute.helper_data import generate_sigma, generate_LRGC, generate_mixed_from_gc
from gcimpute.helper_mask import mask_MCAR as gcimpute_mask_MCAR
from statsmodels.distributions.empirical_distribution import ECDF

from vcimpute.utils import make_triangular_array


def make_complete_data_matrix(n, d, copula_type, seed, **kwargs):
    """
    additional kwargs
    LRGC: scalar valued [rank] and [sigma]
    GC (zhao): no vine_structure
    VC (pyvinecopulib): character [vine_structure]
    """
    if copula_type == 'LRGC':
        assert 'sigma' in kwargs, 'LRGC needs param sigma'
        assert 'rank' in kwargs, 'LRGC needs param rank'
        X, _ = generate_LRGC(
            var_types={'cont': list(range(d))},
            rank=kwargs['rank'],
            sigma=kwargs['sigma'],
            n=n,
            seed=seed
        )
        U = probability_integral_transform(X)
    elif copula_type == 'gaussian' and ('vine_structure' not in kwargs):
        sigma = generate_sigma(p=d)
        X = generate_mixed_from_gc(
            var_types={'cont': list(range(d))},
            n=n,
            sigma=sigma,
            seed=seed
        )
        U = probability_integral_transform(X)
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
        theta = _generate_theta(d * (d - 1) // 2, copula_type, seed)
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


def mask_MCAR(X, pattern, mask_frac, seed, **kwargs):
    n = X.shape[0]
    d = X.shape[1]
    rng = np.random.default_rng(seed)
    if pattern == 'univariate':
        X_mask = np.copy(X)
        miss_idx = rng.choice(range(d))
        is_missing = rng.binomial(n=1, p=mask_frac, size=X_mask.shape[0]).astype(bool)
        X_mask[is_missing, miss_idx] = np.nan
    elif pattern == 'monotone':
        assert 'n_cols' in kwargs, 'monotone missingness pattern needs param n_cols'
        n_cols = kwargs['n_cols']
        n_max_mis = int(n * n_cols * mask_frac)
        X_mask = np.copy(X)
        miss_indices = rng.choice(range(d), n_cols, replace=False)
        obs_coords = range(n)
        for k, _ in enumerate(miss_indices):
            n_rem_miss = n_max_mis - np.sum(np.isnan(X_mask))
            if k != (n_cols - 1):
                n_this_mis = rng.integers(low=1, high=max(1, n_rem_miss) / (n_cols - k) + 1)
            else:
                n_this_mis = n_rem_miss
            is_missing = rng.choice(obs_coords, n_this_mis, replace=False)
            obs_coords = np.setdiff1d(obs_coords, is_missing)
            for j in miss_indices[k:]:
                X_mask[is_missing, j] = np.nan
    elif pattern == 'general':
        X_mask = gcimpute_mask_MCAR(X, mask_fraction=mask_frac)
    else:
        raise NotImplementedError('missingness [pattern] must be one of univariate, monotone, general')
    return X_mask


def _generate_theta(d, copula_type, seed):
    rng = np.random.default_rng(seed)
    assert copula_type in ['gaussian', 'student', 'clayton', 'frank']
    theta = None
    if copula_type in ['gaussian', 'student']:
        theta = 2 * (rng.uniform(size=d) - 0.5)  # [-1,1]
    elif copula_type == 'frank':
        sign = 2 * (rng.binomial(n=1, p=0.5, size=d) - 0.5)  # {0,1}
        theta = sign * rng.uniform(0, 35, size=d)  # [-35,35]
    elif copula_type == 'clayton':
        theta = rng.uniform(1e-10, 28, size=d)  # [1e-10, 28]
    return theta


def probability_integral_transform(X):
    U = np.empty(shape=X.shape)
    for j in range(X.shape[1]):
        U[:, j] = ECDF(X[:, j])(X[:, j])
    return U
