from vcimpute.helper_datagen import make_complete_data_matrix
from vcimpute.utils import bicop_family_map


def test_lrgc():
    X = make_complete_data_matrix(1000, 20, 'gaussian', 42, sigma=10, rank=5)
    assert (X.shape[0] == 1000) and (X.shape[1] == 20)


def test_gc_zhao():
    X = make_complete_data_matrix(1000, 20, 'gaussian', 42)
    assert (X.shape[0] == 1000) and (X.shape[1] == 20)


def test_gc_pyvinecopulib_structure():
    for vine_structure in ['R', 'C', 'D']:
        X = make_complete_data_matrix(1000, 20, 'gaussian', 42, vine_structure=vine_structure)
        assert (X.shape[0] == 1000) and (X.shape[1] == 20)


def test_vc_pyvinecopulib():
    for copula_type in bicop_family_map.keys():
        X = make_complete_data_matrix(1000, 20, copula_type, 42, vine_structure='R')
        assert (X.shape[0] == 1000) and (X.shape[1] == 20)
