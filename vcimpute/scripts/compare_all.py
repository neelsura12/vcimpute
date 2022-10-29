import logging
import os
import time
from functools import partial
from itertools import product

import numpy as np
import pandas as pd
from gcimpute.gaussian_copula import GaussianCopula

from vcimpute.helper_datagen import make_complete_data_matrix, mask_MCAR
from vcimpute.sakuth import MdpFit
from vcimpute.zeisberger import VineCopReg, VineCopFit
from vcimpute.utils import smae_mean, bias

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='compare_all.log',
    format='%(asctime)s  %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)


def profiled_run(X, X_mis, f):
    start = time.process_time_ns()
    X_imp = f(X_mis)
    elapsed = time.process_time_ns() - start
    return smae_mean(X_imp, X, X_mis), bias(X_imp, X), elapsed


def impute(X, X_mis, seed, **kwargs):
    out = {}
    for k, v in kwargs.items():
        out[k] = v
    vine_structure = 'R' if kwargs['vine_structure'] is None else kwargs['vine_structure']
    tagged_models = [
        (GaussianCopula(random_state=seed), 'gcimpute'),
        (MdpFit(bicop_family=kwargs['copula_type'], num_threads=10, seed=seed), f'mdpfit{vine_structure}'),
        (VineCopReg(bicop_family=kwargs['copula_type'], num_threads=10, vine_structure=vine_structure, is_monotone=False, seed=seed), f'copreg{vine_structure}'),
        (VineCopFit(bicop_family=kwargs['copula_type'], num_threads=10, is_monotone=False, seed=seed), 'copfit'),
    ]
    for model, tag in tagged_models:
        logger.info(f'running {tag}')
        smae, bias, elapsed = profiled_run(X, X_mis, model.fit_transform)
        out[f'smae_{tag}'] = smae
        out[f'bias_{tag}'] = bias
        out[f'elapsed_{tag}'] = elapsed
    return out


def run_per_mask(pattern, mask_fraction, X, n, d, copula_type, vine_structure, seed):
    logger.info(f'on pattern {pattern} with mask fraction {mask_fraction:.2f}')
    if pattern == 'general':
        X_mis = mask_MCAR(X, pattern, mask_fraction, seed=seed)
        return impute(
            X,
            X_mis,
            seed,
            copula_type=copula_type,
            vine_structure=vine_structure,
            pattern=pattern,
            mask_fraction=mask_fraction,
            n=n,
            d=d
        )


def run_per_data(X, n, d, seed, copula_type, vine_structure, k):
    logger.info(f'on data n={n} d={d} copula_type={copula_type} vine_structure={vine_structure} k={k}')
    pattern_lst = ['general']
    mask_fraction_lst = np.concatenate([
        np.arange(0.01, 0.08, 0.025)
    ])
    f = partial(run_per_mask, X=X, n=n, d=d, copula_type=copula_type, vine_structure=vine_structure, seed=seed)
    R = 1
    for r in range(R):
        for pattern, mask_fraction in product(pattern_lst, mask_fraction_lst):
            path = f'/Users/nshah/work/vcimpute/output2/copula_{k}_{r}_{pattern}_{str(int(mask_fraction * 100))}.csv'
            if os.path.isfile(path):
                logger.info('skipping: ' + path)
                continue
            out = f(pattern, mask_fraction)
            pd.DataFrame.from_records(out, index=[0]).to_csv(path, index=False)


def run():
    n = 1000
    d_lst = np.arange(5, 21, 5)
    copula_type_lst = ['gaussian']
    vine_structure_lst = [None, 'R']

    seed = 0
    k = 0
    for d in d_lst:
        for copula_type in copula_type_lst:
            for vine_structure in vine_structure_lst:
                if (vine_structure is None) and (copula_type == 'gaussian'):
                    X = make_complete_data_matrix(n, d, copula_type, seed=seed)
                elif (vine_structure is None) and (copula_type != 'gaussian'):
                    continue
                else:
                    X = make_complete_data_matrix(n, d, copula_type, seed=seed, vine_structure=vine_structure)
                run_per_data(X, n, d, seed, copula_type, vine_structure, k)
                k += 1
                seed += 1


if __name__ == '__main__':
    run()
