import numpy as np
import pyvinecopulib as pv

from vcimpute.utils import get_order


def diagonalize_copula(cop1, T2):
    T1 = cop1.matrix
    assert is_diagonal_matrix(T2)

    order1 = get_order(T1)
    order2 = get_order(T2)

    d = T1.shape[0]
    pair_copulas = []
    for i2 in range(d - 1):
        tree = []
        pair_copulas.append(tree)
        for j2 in range(d - i2 - 1):
            j1 = order1.index(order2[j2])
            if T2[i2, j2] in T1[:, j1]:
                i1 = list(T1[:, j1]).index(T2[i2, j2])
            else:
                j1 = order1.index(T2[i2, j2])
                i1 = list(T1[:, j1]).index(order2[j2])
            tree.append(cop1.get_pair_copula(i1, j1))
    cop2 = pv.Vinecop(matrix=T2, pair_copulas=pair_copulas)
    return cop2


def diagonalize_matrix(T1):
    d = T1.shape[0]
    T2 = np.zeros((d, d), dtype=np.uint64)
    forbidden = []
    for j in range(d):
        for i in range(d - j)[::-1]:
            if i == (d - 1):
                T2[d - 1, j] = T1[d - 1, j]
            elif (i == (d - 2)) and (j == 0):
                T2[d - 2, j] = T1[d - 2, j]
            elif (i == (d - j - 1)) and (j != 0):
                T2[d - j - 1, j] = T2[d - j - 1, j - 1]
            else:
                for k in range(d - i - 1):
                    if (T1[d - k - 1, k] == T2[d - j - 1, j]) and (T1[i, k] not in forbidden):
                        T2[i, j] = T1[i, k]
                    elif (T1[i, k] == T2[d - j - 1, j]) and (T1[d - k - 1, k] not in forbidden):
                        T2[i, j] = T1[d - k - 1, k]
            forbidden.append(T2[d - j - 1, j])
    return T2


def diagonalize_matrix2(T1):
    assert is_diagonal_matrix(T1)
    d = T1.shape[0]
    T2 = np.copy(T1)
    m1 = T2[d - 1, 0]
    m2 = T2[d - 2, 0]
    T2[d - 1, 0] = m2
    T2[d - 2, :2] = m1
    r1 = np.copy(T2[:d - 2, 0])
    T2[:d - 2, 0] = T2[:d - 2, 1]
    T2[:d - 2, 1] = r1
    T2 = diagonalize_matrix(T2)
    return T2


def is_diagonal_matrix(T):
    d = T.shape[1]
    is_diagonal = True
    for j in range(d - 1):
        is_diagonal &= T[d - j - 2, j] == T[d - j - 2, j + 1]
    return is_diagonal
