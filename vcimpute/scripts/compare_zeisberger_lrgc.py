# want to measure runtime, smae, bias
import time

import numpy as np
import pandas as pd
from gcimpute.gaussian_copula import GaussianCopula

from vcimpute.helper_datagen import make_complete_data_matrix, mask_MCAR
from vcimpute.utils import smae_mean, bias
from vcimpute.zeisberger import VineCopReg, VineCopFit


def profiled_run(X, X_mis, f):
    start = time.process_time_ns()
    X_imp = f(X_mis)
    elapsed = time.process_time_ns() - start
    return smae_mean(X_imp, X, X_mis), bias(X_imp, X), elapsed


def impute(X, X_mis, seed, is_monotone, **kwargs):
    out = {}
    for k, v in kwargs.items():
        out[k] = v

    tagged_models = [
        (GaussianCopula(random_state=seed), 'gcimpute'),
        (VineCopReg(bicop_family='gaussian', num_threads=10, vine_structure='R', is_monotone=False, seed=seed),
         'copregR'),
        (VineCopReg(bicop_family='gaussian', num_threads=10, vine_structure='C', is_monotone=False, seed=seed),
         'copregC'),
        (VineCopReg(bicop_family='gaussian', num_threads=10, vine_structure='D', is_monotone=False, seed=seed),
         'copregD'),
        (VineCopFit(bicop_family='gaussian', num_threads=10, is_monotone=False, seed=seed), 'copfit')
    ]
    if is_monotone:
        tagged_models.extend([
            (VineCopReg(bicop_family='gaussian', num_threads=10, vine_structure='R', is_monotone=True, seed=seed),
             'copregR_mon'),
            (VineCopReg(bicop_family='gaussian', num_threads=10, vine_structure='C', is_monotone=True, seed=seed),
             'copregC_mon'),
            (VineCopReg(bicop_family='gaussian', num_threads=10, vine_structure='D', is_monotone=True, seed=seed),
             'copregD_mon'),
            (VineCopFit(bicop_family='gaussian', num_threads=10, is_monotone=True, seed=seed), 'copfit_mon')
        ])

    for model, tag in tagged_models:
        smae, bias, elapsed = profiled_run(X, X_mis, model.fit_transform)
        out[f'smae_{tag}'] = smae
        out[f'bias_{tag}'] = bias
        out[f'elapsed_{tag}'] = elapsed

    return out


def run_comparison(X, n, d, rank, sigma, seed, k):
    pattern_lst = ['univariate', 'monotone', 'general']
    mask_fraction_lst = np.concatenate([
        np.arange(0.01, 0.1, 0.01),
        np.arange(0.1, 0.2, 0.02),
        np.arange(0.2, 0.4, 0.05)
    ])
    n_col_max_frac_arr = np.array([0.15, 0.30, 0.45])

    R = 10
    out = []
    for _ in range(R):
        for pattern in pattern_lst:
            for mask_fraction in mask_fraction_lst:
                if pattern == 'monotone':
                    for n_cols in np.unique(np.int64(np.ceil(n_col_max_frac_arr * d))):
                        X_mis = mask_MCAR(X, pattern, mask_fraction, seed=seed, n_cols=n_cols)
                        out.append(impute(
                            X,
                            X_mis,
                            seed,
                            True,
                            pattern=pattern,
                            mask_fraction=mask_fraction,
                            n_cols=n_cols,
                            n=n,
                            d=d,
                            rank=rank,
                            sigma=sigma
                        ))
                else:
                    X_mis = mask_MCAR(X, pattern, mask_fraction, seed=seed)
                    out.append(impute(
                        X,
                        X_mis,
                        seed,
                        False,
                        pattern=pattern,
                        mask_fraction=mask_fraction,
                        n=n,
                        d=d,
                        rank=rank,
                        sigma=sigma
                    ))

    pd.DataFrame(out).to_csv(f'/Users/nshah/work/vcimpute/output/lrgc_{k}.csv')


def run(debug=False):
    n_lst = [1000, 5000]
    d_lst_of_lst = [np.arange(10, 110, 5), np.arange(10, 510, 10)]
    rank_prop_lst = [0.25, 0.5, 0.75]
    sigma_lst = [0.01, 0.1, 0.5]
    if debug:
        n_lst = [1000]
        d_lst_of_lst = [[10, 11, 12]]
        rank_prop_lst = [0.25]
        sigma_lst = [0.1]

    seed = 0
    k = 0
    for n, d_lst in zip(n_lst, d_lst_of_lst):
        for d in d_lst[::-1]:
            seen_ranks = []
            for rank_prop in rank_prop_lst:
                rank = int(np.ceil(rank_prop * d))
                if rank not in seen_ranks:
                    for sigma in sigma_lst:
                        X = make_complete_data_matrix(n, d, 'LRGC', seed=seed, rank=rank, sigma=sigma)
                        run_comparison(X, n, d, rank, sigma, seed, k)
                        k += 1
                        seed += 1
                seen_ranks.append(rank)


if __name__ == '__main__':
    run()
