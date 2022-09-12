import numpy as np
import pytest

from vcimpute.helper_mdp import reindex_monotonic, all_mdps, select_by_mdp


@pytest.fixture
def X_mis():
    return np.array([
        [0, np.nan, np.nan],
        [1, 1, 1],
        [2, np.nan, 2],
        [3, np.nan, 3]
    ])


def test_reindex_monotonic(X_mis):
    new_indices = reindex_monotonic(X_mis)
    assert np.array_equal(new_indices, [0, 2, 1])


def test_all_mdps(X_mis):
    mdps = all_mdps(X_mis)
    assert np.array_equal(mdps, [[False, True, False],
                                 [False, True, True]])


def test_select_by_mdp(X_mis):
    X_sub = select_by_mdp(X_mis, [False, True, False])
    assert X_sub.shape[0] == 2
