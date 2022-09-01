import random

import numpy as np
import pyvinecopulib as pv

from vcimpute.constant import bicop_family_map
from vcimpute.simulator import simulate_orderk
from vcimpute.util import get, make_triangular_array, get_order, vine_structure_to_matrix


class VineCopReg:
    def __init__(self, bicop_families, num_threads, vine_structure):
        family_set = [bicop_family_map[k] for k in bicop_families]
        self.controls = pv.FitControlsVinecop(family_set=family_set, num_threads=num_threads)
        assert vine_structure in ['R', 'C', 'D']
        self.vine_structure = vine_structure

    def fit_transform(self, X_mis):
        d = X_mis.shape[1]
        n_mis = np.sum(np.any(np.isnan(X_mis), axis=0))

        # check monotonic
        for j in range(d):
            this_var_nan = np.isnan(X_mis[:, j])
            assert np.all(np.isnan(X_mis[this_var_nan, (j + 1):])), 'non-monotonic missingness pattern'

        # simulate vine structure for sequential imputation
        structure = None
        if self.vine_structure == 'R':
            structure = _generate_r_vine_structure(d, n_mis)
        elif self.vine_structure == 'C':
            structure = _generate_c_or_d_vine_structure(d, n_mis, pv.CVineStructure)
        elif self.vine_structure == 'D':
            structure = _generate_c_or_d_vine_structure(d, n_mis, pv.DVineStructure)
        assert structure is not None

        # make copula with fixed structure
        pcs = make_triangular_array(d)
        for j in range(d - 1):
            for i in range(d - j - 1):
                pcs[i][j] = pv.Bicop()
        cop = pv.Vinecop(structure=structure, pair_copulas=pcs)

        X_imp = np.copy(X_mis)
        for k in range(n_mis)[::-1]:
            cop.select(X_imp, controls=self.controls)
            x_imp = simulate_orderk(cop, X_imp, k)
            assert not np.any(np.isnan(x_imp)), 'check imputation order'

            x_mis = get(X_imp, cop.order[k])
            is_missing = np.isnan(x_mis)
            x_mis[is_missing] = x_imp[is_missing]

        assert not np.any(np.isnan(X_imp)), 'invalid state, not all values imputed'
        return X_imp


def _generate_r_vine_structure(d, n_mis):
    # simulate Rvine structure
    structure = pv.RVineStructure.simulate(d=d)
    mat = vine_structure_to_matrix(structure)

    # relabel Rvine matrix
    mat2 = np.copy(mat)
    for k in range(n_mis):
        order = get_order(mat2)
        if order[k] != (d - k):
            prev = order[k]
            mat2 = np.where(mat == prev, d - k, mat2)
            mat2 = np.where(mat == d - k, prev, mat2)
        mat = mat2

    # output Rvine structure
    return pv.RVineStructure(mat2)


def _generate_c_or_d_vine_structure(d, n_mis, struct_fun):
    rest_indices = list(range(1, d - n_mis + 1))
    random.shuffle(rest_indices)
    mis_indices = list(range(d - n_mis + 1, d + 1))[::-1]
    return struct_fun(order=mis_indices + rest_indices)
