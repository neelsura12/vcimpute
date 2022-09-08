import random

import numpy as np
import pyvinecopulib as pv

from vcimpute.utils import get_order, vine_structure_to_matrix


def relabel_vine_mat(T, old_to_new):
    Tnew = np.copy(T)
    for old, new in old_to_new.items():
        Tnew = np.where(T == old, new, Tnew)
    return Tnew


def natural_order_mat(T):
    structure = pv.RVineStructure(T)
    d = T.shape[0]
    T2 = np.zeros(shape=(d, d), dtype=np.uint64)
    for j in range(d - 1):
        for i in range(d - j - 1):
            T2[i, j] = structure.struct_array(i, j, natural_order=True)
    for j in range(d):
        T2[d - j - 1, j] = j + 1
    return T2


# TODO: generify
def generate_r_vine_structure(d, n_mis):
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


def generate_c_or_d_vine_structure(d, n_mis, struct_fun):
    rest_indices = list(range(1, d - n_mis + 1))
    random.shuffle(rest_indices)
    mis_indices = list(range(d - n_mis + 1, d + 1))[::-1]
    return struct_fun(order=mis_indices + rest_indices)
