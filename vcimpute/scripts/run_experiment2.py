import pickle
import time

import numpy as np
from gcimpute.gaussian_copula import GaussianCopula
from gcimpute.helper_evaluation import get_smae
from joblib import Parallel, delayed

from vcimpute.utils import bias
from vcimpute.helper_datagen import make_complete_data_matrix
from vcimpute.helper_datagen import mask_MCAR
from vcimpute.sakuth import VineMdpFit
from vcimpute.zeisberger import VineCopFit, VineCopReg


from scipy.stats import kendalltau

def profiled_run(seed, copula_type):
    n = 1000
    d = 10
    mask_frac = 0.1
    num_threads = 10

    model_lst = [
        ('gcimpute', GaussianCopula()),
        ('mdpfit', VineMdpFit(copula_type, num_threads, seed)),
        ('copfit', VineCopFit(copula_type, num_threads, True, seed)),
        ('copreg', VineCopReg(copula_type, num_threads, 'R', True, seed)),
    ]

    # make data
    X = make_complete_data_matrix(n, d, copula_type, seed, vine_structure='R')

    # mask data
    X_mask = mask_MCAR(X, 'univariate', mask_frac, seed)

    # calculate top 3 kendall tau
    idx = np.where(np.any(np.isnan(X_mask), axis=0))[0].item()
    out = []
    for j in set(range(X.shape[1])).difference({idx}):
        out.append(kendalltau(X[:, j], X[:, idx]).correlation)
    max_kt = np.amax(np.abs(np.array(out)))
    mean_kt = np.mean(np.abs(np.array(out)))

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
            max_kt,
            mean_kt,
            get_smae(X_imp, X, X_mask)[idx],
            elapsed,
            bias(X_imp, X),
        ))
    return out


if __name__ == '__main__':
    for copula_type in ['clayton', 'frank', 'gaussian']:
        out = Parallel(n_jobs=-1)(delayed(profiled_run)(seed, copula_type) for seed in range(5000))
        pickle.dump(out, open(f'experiment2_{copula_type}.pkl', 'wb'))
