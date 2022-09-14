import numpy as np
import pyvinecopulib as pv


def diagonalize_copula(cop1, var, bicop_family):
    T1 = cop1.matrix
    d = T1.shape[0]
    ced, cing, param = get_ced_cing(T1, cop1)

    T2 = diagonalize_matrix(T1, var)

    pair_copulas = []
    for t in range(d - 1):
        cur = []
        pair_copulas.append(cur)
        for e in range(d - 1 - t):
            cur.append(
                pv.Bicop(
                    family=bicop_family,
                    parameters=param[ced.index(sorted((T2[d - 1 - e, e], T2[t, e])))]
                )
            )
    cop2 = pv.Vinecop(matrix=T2, pair_copulas=pair_copulas)
    return cop2


def diagonalize_matrix(T1, a):
    d = T1.shape[1]
    if a == T1[d - 1, 0]:
        return T1
    assert a == T1[d - 2, 0], f'cannot be diagonalized with {a}'

    T2 = np.zeros(shape=T1.shape, dtype=np.uint64)
    T2[d - 1, 0] = a
    order = [a]

    ced, cing, _ = get_ced_cing(T1)
    for j in range(d - 1):
        for i in range(d - j - 1):
            T2[i, j] = find(T2[d - j - 1, j], i, ced, cing)

        remove_idx = [i for i, c in enumerate(ced) for k in order if k in c]
        keep_idx = set(range(len(ced))).difference(set(remove_idx))

        ced = [ced[i] for i in keep_idx]
        cing = [cing[i] for i in keep_idx]

        T2[d - j - 2, j + 1] = T2[d - j - 2, j]
        order.append(T2[d - j - 2, j + 1])
    return T2


def get_ced_cing(T, cop=None):
    d = T.shape[1]
    cing = []
    ced = []
    param = []
    for j in range(d):
        for i1 in range(d - j - 1):
            ced.append(sorted((T[i1, j], T[d - j - 1, j])))
            tmp = []
            for i2 in range(i1):
                tmp.append(T[i2, j])
            cing.append(sorted(tmp))
            if cop is not None:
                param.append(cop.get_parameters(i1, j))
    return ced, cing, param


def find(a, cing_len, ced, cing):
    out = [i for i in range(len(ced)) if a in ced[i]]
    matched = False
    for i in out:
        if len(cing[i]) == cing_len:
            matched = True
            break
    assert matched, f'bad argument, a={a}, cing_len={cing_len}'
    return ced[i][1] if ced[i][0] == a else ced[i][0]
