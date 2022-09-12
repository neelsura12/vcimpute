import numpy as np
import pyvinecopulib as pv

from vcimpute.utils import get_order


def generate_r_vine_structure(miss_vars, obs_vars):
    d = len(miss_vars) + len(obs_vars)

    # simulate
    structure_random = pv.RVineStructure.simulate(d=d)
    T_random = vine_structure_to_matrix(structure_random)

    old_order = get_order(T_random)
    old_to_new = {}
    for i, var_mis in enumerate(miss_vars):
        old_to_new[old_order[i]] = var_mis
    for i, var_obs in zip(range(len(miss_vars), d), obs_vars):
        old_to_new[old_order[i]] = var_obs

    # relabel
    T_ordered = relabel_vine_matrix(T_random, old_to_new)
    structure_ordered = pv.RVineStructure(T_ordered)

    return structure_ordered


def vine_structure_to_matrix(structure):
    d = len(structure.order)
    T = np.zeros(shape=(d, d), dtype=np.uint64)
    for j in range(d):
        for i in range(d - j - 1):
            T[j, i] = structure.struct_array(j, i)
        T[d - j - 1, j] = structure.order[j]
    return T


def relabel_vine_matrix(T_old, old_to_new):
    T_new = np.copy(T_old)
    for old, new in old_to_new.items():
        T_new = np.where(T_old == old, new, T_new)
    return T_new
