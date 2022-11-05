import pickle
import time

import numpy as np
from gcimpute.gaussian_copula import GaussianCopula
from gcimpute.helper_evaluation import get_smae
from joblib import Parallel, delayed

from vcimpute.helper_datagen import make_complete_data_matrix, mask_MCAR
from vcimpute.helper_mdp import all_mdps, count_missing_by_row
from vcimpute.sakuth import MdpFit
from vcimpute.utils import bias
from vcimpute.zeisberger import VineCopFit, VineCopReg


def profiled_run(seed):
    n = 1000
    d = 5
    mask_frac = 0.1
    num_threads = 10
    copula_type = 'gaussian'
    vine_structure = 'R'

    model_lst = [
        ('gcimpute', GaussianCopula()),
        ('mdpfit', MdpFit(copula_type, num_threads, seed)),
        ('copfit', VineCopFit(copula_type, num_threads, False, seed)),
        ('copreg', VineCopReg(copula_type, num_threads, vine_structure, False, seed)),
    ]

    # make data
    X = make_complete_data_matrix(n, d, 'gaussian', seed)

    # mask data
    X_mask = mask_MCAR(X, 'general', mask_frac, seed)

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
            np.sum(count_missing_by_row(X_mask) == 0),  # complete cases
            get_smae(X_imp, X, X_mask),
            elapsed,
            bias(X_imp, X),
        ))
    return out


if __name__ == '__main__':
    out = Parallel(n_jobs=-1)(delayed(profiled_run)(seed) for seed in range(10))
    pickle.dump(out, open(f'experiment3_general.pkl', 'wb'))
