import numpy as np
import pytest

from vcimpute.helper_choicetree import make_tree, is_in_tree


@pytest.fixture
def T_3d():
    return np.array([
        [2, 2, 2],
        [3, 3, 0],
        [1, 0, 0]
    ], dtype=np.uint64)


@pytest.fixture
def T_5d():
    return np.array([
        [3, 2, 3, 3, 3],
        [2, 3, 2, 2, 0],
        [4, 4, 4, 0, 0],
        [1, 1, 0, 0, 0],
        [5, 0, 0, 0, 0]
    ], dtype=np.uint64)


def test_3d(T_3d):
    root = make_tree(T_3d)
    assert is_in_tree(root, [1, 3])
    assert not is_in_tree(root, [1, 3, 2])


def test_5d(T_5d):
    root = make_tree(T_5d)
    assert is_in_tree(root, [5, 1, 2, 3])
    assert is_in_tree(root, [5, 4, 2, 3])
