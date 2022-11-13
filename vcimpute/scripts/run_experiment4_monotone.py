import pickle
import time

import numpy as np
from gcimpute.gaussian_copula import GaussianCopula
from gcimpute.helper_evaluation import get_smae
from joblib import Parallel, delayed

from vcimpute.helper_datagen import make_complete_data_matrix, mask_MCAR
from vcimpute.helper_mdp import all_mdps, count_missing_by_col, count_missing_by_row
from vcimpute.sakuth import VineMdpFit
from vcimpute.utils import bias
from vcimpute.zeisberger import VineCopFit, VineCopReg


def profiled_run(seed, d, mask_frac, n_cols):
    n = 1000
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
            np.sum(count_missing_by_row(X_mask) == 0),
            count_missing_by_col(X_mask),
            get_smae(X_imp, X, X_mask),
            elapsed,
            bias(X_imp, X),
        ))
    pickle.dump(out, open(f'experiment4_monotone_{d}_{mask_frac}_{n_cols}_{seed}.pkl', 'wb'))


if __name__ == '__main__':

    params = [
        (100, 0.48, 21),
        (105, 0.48, 22),
        (106, 0.48, 22),
        (108, 0.49, 22),
        (110, 0.48, 23),
        (111, 0.48, 23),
        (113, 0.49, 23),
        (120, 0.48, 25),
        (129, 0.5, 26),
        (132, 0.49, 27),
        (137, 0.49, 28),
        (140, 0.48, 29),
        (141, 0.49, 29),
        (149, 0.5, 30),
        (157, 0.49, 32),
        (159, 0.5, 32),
        (161, 0.49, 33),
        (162, 0.49, 33),
        (165, 0.49, 34),
        (172, 0.49, 35),
        (184, 0.5, 37),
        (185, 0.49, 38),
        (188, 0.49, 38),
        (193, 0.49, 39),
        (199, 0.5, 40)
    ]
    out = Parallel(n_jobs=-1)(delayed(profiled_run)(0, d, mask_frac, n_cols) for d, mask_frac, n_cols in params)
