import logging
import time
from functools import partial
from itertools import product

import numpy as np
import pandas as pd
from gcimpute.gaussian_copula import GaussianCopula

from vcimpute.helper_datagen import make_complete_data_matrix, mask_MCAR
from vcimpute.utils import smae_mean, bias
from vcimpute.zeisberger import VineCopReg, VineCopFit

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='compare_zeisberger_lrgc.log',
    format='%(asctime)s  %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)


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
    ]
    if is_monotone:
        tagged_models.extend([
            (VineCopReg(bicop_family='gaussian', num_threads=10, vine_structure='R', is_monotone=True, seed=seed), 'copregR_mon'),
            (VineCopFit(bicop_family='gaussian', num_threads=10, is_monotone=True, seed=seed), 'copfit_mon')
        ])
    else:
        tagged_models.extend([
            (VineCopReg(bicop_family='gaussian', num_threads=10, vine_structure='R', is_monotone=False, seed=seed), 'copregR'),
            (VineCopFit(bicop_family='gaussian', num_threads=10, is_monotone=False, seed=seed), 'copfit')
        ])
    for model, tag in tagged_models:
        logger.info(f'running {tag}')
        smae, bias, elapsed = profiled_run(X, X_mis, model.fit_transform)
        out[f'smae_{tag}'] = smae
        out[f'bias_{tag}'] = bias
        out[f'elapsed_{tag}'] = elapsed

    return out


def run_per_mask(pattern, mask_fraction, X, n, d, rank, sigma, seed):
    logger.info(f'on pattern {pattern} with mask fraction {mask_fraction:.2f}')
    if pattern == 'monotone':
        n_cols = int(np.ceil(0.3 * d))
        X_mis = mask_MCAR(X, pattern, mask_fraction, seed=seed, n_cols=n_cols)
        return impute(
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
        )
    else:
        X_mis = mask_MCAR(X, pattern, mask_fraction, seed=seed)
        return impute(
            X,
            X_mis,
            seed,
            pattern == 'univariate',
            pattern=pattern,
            mask_fraction=mask_fraction,
            n=n,
            d=d,
            rank=rank,
            sigma=sigma
        )


def run_per_data(X, n, d, rank, sigma, seed, k):
    logger.info(f'on data n={n} d={d} rank={rank} sigma={sigma:.2f}')
    pattern_lst = ['univariate', 'monotone', 'general']
    mask_fraction_lst = np.concatenate([
        np.arange(0.05, 0.1, 0.01),
        np.arange(0.1, 0.2, 0.05),
        [0.2]
    ])
    f = partial(run_per_mask, X=X, n=n, d=d, rank=rank, sigma=sigma, seed=seed)
    R = 10
    for r in range(R):
        for pattern, mask_fraction in product(pattern_lst, mask_fraction_lst):
            (pd.DataFrame(f(pattern, mask_fraction)).to_csv(
                f'/Users/nshah/work/vcimpute/output/lrgc_{k}_{r}_{pattern}_{str(int(mask_fraction * 100))}.csv',
                index=False))


def run():
    n = 1000
    d_lst = np.arange(10, 110, 5)
    rank_prop_lst = [0.25, 0.5, 0.75]
    sigma_lst = [0.01, 0.1]

    seed = 0
    k = 0
    for d in d_lst[::-1]:
        seen_ranks = []
        for rank_prop in rank_prop_lst:
            rank = int(np.ceil(rank_prop * d))
            if rank not in seen_ranks:
                for sigma in sigma_lst:
                    X = make_complete_data_matrix(n, d, 'LRGC', seed=seed, rank=rank, sigma=sigma)
                    run_per_data(X, n, d, rank, sigma, seed, k)
                    k += 1
                    seed += 1
            seen_ranks.append(rank)


if __name__ == '__main__':
    run()
