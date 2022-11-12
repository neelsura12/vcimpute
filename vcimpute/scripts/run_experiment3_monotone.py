import pickle
import time

import numpy as np
from gcimpute.gaussian_copula import GaussianCopula
from gcimpute.helper_evaluation import get_smae
from joblib import Parallel, delayed

from vcimpute.helper_datagen import make_complete_data_matrix, mask_MCAR
from vcimpute.helper_mdp import all_mdps, count_missing_by_row, count_missing_by_col
from vcimpute.sakuth import VineMdpFit
from vcimpute.utils import bias
from vcimpute.zeisberger import VineCopFit, VineCopReg


def profiled_run(seed, n_cols):
    n = 1000
    d = 10
    mask_frac = 0.1
    num_threads = 10
    copula_type = 'gaussian'
    vine_structure = 'R'

    model_lst = [
        ('gcimpute', GaussianCopula()),
        ('mdpfit', VineMdpFit(copula_type, num_threads, seed)),
        ('copfit', VineCopFit(copula_type, num_threads, True, seed)),
        ('copreg', VineCopReg(copula_type, num_threads, vine_structure, True, seed)),
    ]

    # make data
    X = make_complete_data_matrix(n, d, 'gaussian', seed)

    # mask data
    X_mask = mask_MCAR(X, 'monotone', mask_frac, seed, n_cols=n_cols)

    out = []
    for tag, model in model_lst:
        # impute
        start = time.process_time_ns()
        X_imp = model.fit_transform(X_mask)
        elapsed = time.process_time_ns() - start

        # score
        out.append((
            tag,
            seed,
            getattr(model, 'n_fits', 1),
            getattr(model, 'n_sims', 1),
            len(all_mdps(X_mask)),
            n_cols,
            np.sum(count_missing_by_row(X_mask) == 0),  # complete cases
            count_missing_by_col(X_mask),
            get_smae(X_imp, X, X_mask),
            elapsed,
            bias(X_imp, X),
        ))
    return out


if __name__ == '__main__':
    import os
    for i in range(10):
        if os.path.isfile(f'experiment3_monotone_{i}.pkl'):
            print('skipping', i)
            continue
        out = Parallel(n_jobs=4)(delayed(profiled_run)(seed, n_cols) for seed in range(100*i, 100*(i+1)) for n_cols in range(1, 10))
        pickle.dump(out, open(f'experiment3_monotone_{i}.pkl', 'wb'))
