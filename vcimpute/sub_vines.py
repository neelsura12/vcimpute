from copy import deepcopy

import numpy as np

from vcimpute.util import make_triangular_array


def remove_column(T_in, pair_copulas_in, var_mis, j):
    """
    remove column if var_mis in cond. set +
    remove all entries of that column's diagonal var
    """
    d = T_in.shape[0]
    if var_mis in T_in[:d - j - 1, j]:
        T_tmp = deepcopy(T_in)
        var_diag = T_tmp[d - j - 1, j]
        T_tmp[:, j] = 0
        T_tmp = np.where(T_tmp == var_diag, 0, T_tmp)
        return downsize_copula(T_in, pair_copulas_in, T_tmp)
    else:
        return T_in, pair_copulas_in


def remove_inbetween(T_in, pair_copulas_in, var_mis, j):
    """
    identify all vars between the diagonal and var_mis in the col j
    delete all columns with those vars in the diagonal
    delete all entries with those vars
    """
    d = T_in.shape[0]
    if var_mis in T_in[:d - j - 2, j]:
        T_tmp = deepcopy(T_in)
        k = np.where(T_tmp[:d - j - 1, j] == var_mis)[0].item()
        T_tmp[(k + 1):d - j - 1, j] = 0
        order = [T_tmp[d - j - 1, j] for j in range(d)]
        for var_del in T_tmp[(k + 1):d - j - 1, j]:
            T_tmp[:, order.index(var_del)] = 0
            T_tmp = np.where(T_tmp == var_del, 0, T_tmp)
        return downsize_copula(T_in, pair_copulas_in, T_tmp)
    else:
        return T_in, pair_copulas_in


def downsize_copula(T, pair_copulas_in, T_tmp):
    if np.all(T == T_tmp):
        return T, pair_copulas_in

    d = T.shape[0]
    d2 = np.amax(np.count_nonzero(T_tmp, axis=0))
    i_tmp_lst, j_tmp_lst = np.where(T_tmp != 0)
    ax0_order = np.argsort(j_tmp_lst)
    i_tmp_lst = i_tmp_lst[ax0_order]
    j_tmp_lst = j_tmp_lst[ax0_order]

    assert len(j_tmp_lst) == len(i_tmp_lst) == d * (d - 1) // 2

    T_out = np.zeros(shape=(d2, d2), dtype=np.uint64)
    pair_copulas_out = make_triangular_array(d2)
    i2, j2 = 0, 0
    for i_tmp, j_tmp in zip(i_tmp_lst, j_tmp_lst):
        if i2 > d2 - j2 - 1:
            j2 += 1
            i2 = 0
        T_out[i2, j2] = T_tmp[i_tmp, j_tmp]
        if i2 != d2 - j2 - 1:
            pair_copulas_out[i2][j2] = pair_copulas_in[i_tmp][j_tmp]
        i2 += 1
    return T_out, pair_copulas_out
