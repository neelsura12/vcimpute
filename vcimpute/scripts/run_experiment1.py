import pickle
import time

import numpy as np
from gcimpute.gaussian_copula import GaussianCopula
from gcimpute.helper_evaluation import get_smae
from joblib import Parallel, delayed

from vcimpute.utils import bias
from vcimpute.helper_datagen import make_complete_data_matrix
from vcimpute.helper_datagen import mask_MCAR
from vcimpute.sakuth import MdpFit
from vcimpute.zeisberger import VineCopFit, VineCopReg


def profiled_run(seed):
    n = 1000
    d = 10
    mask_frac = 0.1
    num_threads = 10
    model_lst = [
        ('gcimpute', GaussianCopula()),
        ('mdpfit', MdpFit('gaussian', num_threads, seed)),
        ('copfit', VineCopFit('gaussian', num_threads, True, seed)),
        ('copreg', VineCopReg('gaussian', num_threads, 'R', True, seed)),
    ]

    # make data
    rho_hat = 1
    dat1 = make_complete_data_matrix(n, d, 'gaussian', seed, vine_structure='R')
    dat2 = make_complete_data_matrix(n, d, 'gaussian', seed, sigma=np.corrcoef(dat1.T))
    rho_hat = np.linalg.norm(np.corrcoef(dat2.T) - np.corrcoef(dat1.T)) / np.linalg.norm(np.corrcoef(dat1.T))

    # mask data
    dat1_mask = mask_MCAR(dat1, 'univariate', mask_frac, seed)
    idx = np.where(np.any(np.isnan(dat1_mask), axis=0))[0].item()
    dat2_mask = np.copy(dat2)
    dat2_mask[np.isnan(dat1_mask)[:, idx], idx] = np.nan

    out = []
    for tag, model in model_lst:
        # impute
        start = time.process_time_ns()
        dat1_imp = model.fit_transform(dat1_mask)
        dat2_imp = model.fit_transform(dat2_mask)
        elapsed = time.process_time_ns() - start

        # score
        out.append((
            tag,
            seed,
            get_smae(dat1_imp, dat1, dat1_mask)[idx],
            get_smae(dat2_imp, dat2, dat2_mask)[idx],
            elapsed,
            bias(dat1_imp, dat1),
            bias(dat2_imp, dat2)
        ))
    return out


if __name__ == '__main__':
    out = Parallel(n_jobs=-1)(delayed(profiled_run)(seed) for seed in range(100))
    pickle.dump(out, open('../../data/experiment1.pkl', 'wb'))
