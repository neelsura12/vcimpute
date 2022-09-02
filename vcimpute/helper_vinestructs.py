import random

import numpy as np
import pyvinecopulib as pv

from vcimpute.utils import get_order, vine_structure_to_matrix


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
