import os
import pickle
import time

import numpy as np
from gcimpute.gaussian_copula import GaussianCopula
from gcimpute.helper_evaluation import get_smae
from joblib import Parallel, delayed

from vcimpute.helper_datagen import make_complete_data_matrix, mask_MCAR
from vcimpute.helper_mdp import all_mdps
from vcimpute.sakuth import VineMdpFit
from vcimpute.utils import bias
from vcimpute.zeisberger import VineCopFit, VineCopReg


def profiled_run(seed, d):
    if os.path.isfile(f'/Users/nshah/work/vcimpute/data/experiment4_univariate_single/experiment4_univariate_{d}_{seed}.pkl'):
        return

    n = 1000
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
    X_mask = mask_MCAR(X, 'univariate', mask_frac, seed)

    idx_mis = np.where(np.any(np.isnan(X_mask), axis=0))[0].item()

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
            get_smae(X_imp, X, X_mask)[idx_mis],
            elapsed,
            bias(X_imp, X),
        ))
    pickle.dump(out, open(f'/Users/nshah/work/vcimpute/data/experiment4_univariate_single/experiment4_univariate_{d}_{seed}.pkl', 'wb'))


if __name__ == '__main__':
    for seed in range(100):
        Parallel(n_jobs=8)(delayed(profiled_run)(seed=seed, d=d) for d in range(20, 101, 10))
