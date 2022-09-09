from collections import deque
from copy import deepcopy

import numpy as np
import pandas as pd

from vcimpute.utils import get_order
from vcimpute.utils import make_triangular_array, is_leaf_in_all_subtrees


def find_subvine_structures(T, pcs, var_mis):
    unexplored = deque([(T, pcs)])
    accepted = []
    while len(unexplored) > 0:
        T_cur, pcs_cur = unexplored.pop()
        d_cur = T_cur.shape[0]
        for func in [remove_inbetween, remove_column]:
            T_cand, pcs_cand = func(T_cur, pcs_cur, var_mis, 0)
            d_cand = T_cand.shape[0]
            if is_leaf_in_all_subtrees(T_cand, var_mis):
                if not np.any(list(map(lambda x: np.array_equal(x[0], T_cand), accepted))):
                    accepted.append((T_cand, pcs_cand))
            elif 1 < d_cand < d_cur:
                unexplored.append((T_cand, pcs_cand))
    return accepted


def remove_inbetween(T_in, pcs_in, var_mis, j):
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
        for var_del in T_in[(k + 1):d - j - 1, j]:
            T_tmp[:, order.index(var_del)] = 0
            T_tmp = np.where(T_tmp == var_del, 0, T_tmp)
        return downsize_copula(T_in, pcs_in, T_tmp)
    else:
        return T_in, pcs_in


def remove_column(T_in, pcs_in, var_mis, j):
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
        return downsize_copula(T_in, pcs_in, T_tmp)
    else:
        return T_in, pcs_in


def remove_var(T_in, pcs_in, var_mis):
    """
    remove column
    remove all entries of that column's diagonal var
    """
    order = get_order(T_in)
    j = order.index(var_mis)
    T_tmp = deepcopy(T_in)
    T_tmp[:, j] = 0
    T_tmp = np.where(T_tmp == var_mis, 0, T_tmp)
    return downsize_copula(T_in, pcs_in, T_tmp)


def downsize_copula(T_in, pcs_in, T_cand):
    if np.all(T_in == T_cand):
        return T_in, pcs_in

    d2 = np.amax(np.count_nonzero(T_cand, axis=0))
    ij_tmp = pd.DataFrame(np.where(T_cand != 0)).T.sort_values(by=[1, 0]).values
    assert ij_tmp.shape[0] == (d2 ** 2 - d2 * (d2 - 1) // 2)

    T_out = np.zeros(shape=(d2, d2), dtype=np.uint64)
    pcs_out = make_triangular_array(d2)
    i2, j2 = 0, 0
    for i_tmp, j_tmp in ij_tmp:
        if i2 > d2 - j2 - 1:
            j2 += 1
            i2 = 0
        T_out[i2, j2] = T_cand[i_tmp, j_tmp]
        if i2 != d2 - j2 - 1:
            pcs_out[i2][j2] = pcs_in[i_tmp][j_tmp]
        i2 += 1
    return T_out, pcs_out
