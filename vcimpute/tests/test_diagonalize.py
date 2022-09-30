import numpy as np
import pytest
import pyvinecopulib as pv

from vcimpute.helper_diagonalize import diagonalize_matrix, is_diagonal_matrix, diagonalize_copula


@pytest.fixture
def non_diagonal_matrix():
    return np.array([[2, 3, 4, 5, 6, 6],
                     [3, 4, 5, 6, 5, 0],
                     [4, 5, 6, 4, 0, 0],
                     [5, 6, 3, 0, 0, 0],
                     [6, 2, 0, 0, 0, 0],
                     [1, 0, 0, 0, 0, 0]], dtype=np.uint64)


@pytest.fixture
def diagonal_matrix():
    return np.array([[2, 5, 3, 4, 4, 4],
                     [3, 4, 4, 3, 3, 0],
                     [4, 3, 5, 5, 0, 0],
                     [5, 2, 2, 0, 0, 0],
                     [6, 6, 0, 0, 0, 0],
                     [1, 0, 0, 0, 0, 0]], dtype=np.uint64)


def test_is_diagonal(diagonal_matrix, non_diagonal_matrix):
    assert not is_diagonal_matrix(non_diagonal_matrix)
    assert is_diagonal_matrix(diagonal_matrix)


def test_diagonalize_matrix(diagonal_matrix, non_diagonal_matrix):
    assert np.array_equal(diagonal_matrix, diagonalize_matrix(non_diagonal_matrix))


def test_diagonalize_copula1():
    pair_copulas = [
        [pv.Bicop(family=pv.BicopFamily.gaussian, parameters=[[0.25]]),
         pv.Bicop(family=pv.BicopFamily.gaussian, parameters=[[0.5]])],
        [pv.Bicop(family=pv.BicopFamily.gaussian, parameters=[[0.1]])]
    ]
    T = np.array([
        [2, 3, 3],
        [3, 2, 0],
        [1, 0, 0],
    ], dtype=np.uint64)
    cop1 = pv.Vinecop(T, pair_copulas)
    cop2 = diagonalize_copula(cop1)
    assert cop2.get_pair_copula(0, 0).parameters[0][0] == 0.25
    assert cop2.get_pair_copula(0, 1).parameters[0][0] == 0.5
    assert cop2.get_pair_copula(1, 0).parameters[0][0] == 0.1


def test_diagonalize_copula2():
    pair_copulas = [
        [pv.Bicop(family=pv.BicopFamily.gaussian, parameters=[[0.25]]),
         pv.Bicop(family=pv.BicopFamily.gaussian, parameters=[[0.5]]),
         pv.Bicop(family=pv.BicopFamily.gaussian, parameters=[[0.75]])],
        [pv.Bicop(family=pv.BicopFamily.gaussian, parameters=[[0.33]]),
         pv.Bicop(family=pv.BicopFamily.gaussian, parameters=[[0.66]])],
        [pv.Bicop(family=pv.BicopFamily.gaussian, parameters=[[0.1]])]
    ]
    T = np.array([
        [2, 3, 4, 4],
        [3, 4, 3, 0],
        [4, 2, 0, 0],
        [1, 0, 0, 0]
    ], dtype=np.uint64)
    cop1 = pv.Vinecop(T, pair_copulas)
    cop2 = diagonalize_copula(cop1)
    assert cop2.get_pair_copula(0, 1).parameters[0][0] == 0.75
    assert cop2.get_pair_copula(0, 2).parameters[0][0] == 0.5
