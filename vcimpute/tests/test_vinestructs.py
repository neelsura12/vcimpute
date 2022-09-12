import numpy as np

from vcimpute.helper_vinestructs import generate_r_vine_structure, relabel_vine_matrix


def test_rvinestruct_gen():
    miss_vars, obs_vars = [2, 4], [1, 3, 5]
    structure = generate_r_vine_structure(miss_vars, obs_vars)
    assert np.array_equal(structure.order, np.concatenate([miss_vars, obs_vars]))


def test_relabel_vine_matrix():
    T_old = np.array([[1, 2, 3], [2, 3, 1], [3, 2, 1]])
    exp_T_new = np.array([[4, 5, 6], [5, 6, 4], [6, 5, 4]])
    T_new = relabel_vine_matrix(T_old, {1: 4, 2: 5, 3: 6})
    assert np.array_equal(T_new, exp_T_new)
