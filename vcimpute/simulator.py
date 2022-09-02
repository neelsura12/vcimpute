import numpy as np

from vcimpute.utils import get, vfunc, find


def simulate_order0(cop, U):
    d = U.shape[1]
    T = cop.matrix

    # forward
    V1 = np.empty(shape=(d, d), dtype=object)
    V2 = np.empty(shape=(d, d), dtype=object)
    W = np.empty(shape=(d, d), dtype=object)
    D1 = np.empty(shape=(d, d), dtype=object)
    D2 = np.empty(shape=(d, d), dtype=object)

    for j in range(d - 1)[::-1]:
        for i in range(d - j - 1):
            var1 = cop.order[j]
            var2 = T[i, j]
            W[i, j] = ','.join(list(map(str, sorted(T[:i, j]))))
            if W[i, j] == '':
                arg1 = get(U, var1)
                arg2 = get(U, var2)
                V1[i, j] = vfunc(cop.get_pair_copula(i, j).hfunc2, arg1, arg2)
                V2[i, j] = vfunc(cop.get_pair_copula(i, j).hfunc1, arg1, arg2)
                D1[i, j] = f'{var1}|{var2}'
                D2[i, j] = f'{var2}|{var1}'
            else:
                arg1, arg2 = None, None
                key1 = f'{var1}|{W[i, j]}'
                key2 = f'{var2}|{W[i, j]}'
                for D, V in zip([D1, D2], [V1, V2]):
                    coord = find(D, key1)
                    if (arg1 is None) and (coord is not None):
                        arg1 = V[coord]
                    coord = find(D, key2)
                    if (arg2 is None) and (coord is not None):
                        arg2 = V[coord]
                if (arg1 is not None) and (arg2 is not None):
                    V1[i, j] = vfunc(cop.get_pair_copula(i, j).hfunc2, arg1, arg2)
                    V2[i, j] = vfunc(cop.get_pair_copula(i, j).hfunc1, arg1, arg2)
                    D1[i, j] = f'{var1}|' + ','.join(map(str, sorted(list(map(int, W[i, j].split(','))) + [var2])))
                    D2[i, j] = f'{var2}|' + ','.join(map(str, sorted(list(map(int, W[i, j].split(','))) + [var1])))

    # backward
    w = np.random.uniform(size=U.shape[0])
    for i in range(d - 1)[::-1]:
        if W[i, 0] == '':
            arg2 = get(U, T[i, 0])
        else:
            arg2 = None
            key2 = f'{T[i, 0]}|{W[i, 0]}'
            for D, V in zip([D1, D2], [V1, V2]):
                coord = find(D, key2)
                if (arg2 is None) and (coord is not None):
                    arg2 = V[coord]
        assert arg2 is not None
        w = vfunc(cop.get_pair_copula(i, 0).hinv2, w, arg2)

    return w


def simulate_orderk(cop, U, k):
    d = U.shape[1]
    T = cop.matrix

    # forward
    V1 = np.empty(shape=(d, d), dtype=object)
    V2 = np.empty(shape=(d, d), dtype=object)
    W = np.empty(shape=(d, d), dtype=object)
    D1 = np.empty(shape=(d, d), dtype=object)
    D2 = np.empty(shape=(d, d), dtype=object)

    for j in range(k, d - 1)[::-1]:
        for i in range(d - j - 1):
            var1 = cop.order[j]
            var2 = T[i, j]
            W[i, j] = ','.join(list(map(str, sorted(T[:i, j]))))
            if W[i, j] == '':
                arg1 = get(U, var1)
                arg2 = get(U, var2)
                V1[i, j] = vfunc(cop.get_pair_copula(i, j).hfunc2, arg1, arg2)
                V2[i, j] = vfunc(cop.get_pair_copula(i, j).hfunc1, arg1, arg2)
                D1[i, j] = f'{var1}|{var2}'
                D2[i, j] = f'{var2}|{var1}'
            else:
                arg1, arg2 = None, None
                key1 = f'{var1}|{W[i, j]}'
                key2 = f'{var2}|{W[i, j]}'
                for D, V in zip([D1, D2], [V1, V2]):
                    coord = find(D, key1)
                    if (arg1 is None) and (coord is not None):
                        arg1 = V[coord]
                    coord = find(D, key2)
                    if (arg2 is None) and (coord is not None):
                        arg2 = V[coord]
                if (arg1 is not None) and (arg2 is not None):
                    V1[i, j] = vfunc(cop.get_pair_copula(i, j).hfunc2, arg1, arg2)
                    V2[i, j] = vfunc(cop.get_pair_copula(i, j).hfunc1, arg1, arg2)
                    D1[i, j] = f'{var1}|' + ','.join(map(str, sorted(list(map(int, W[i, j].split(','))) + [var2])))
                    D2[i, j] = f'{var2}|' + ','.join(map(str, sorted(list(map(int, W[i, j].split(','))) + [var1])))

    # backward
    w = np.random.uniform(size=U.shape[0])
    for i in range(d - k - 1)[::-1]:
        if W[i, k] == '':
            arg2 = get(U, T[i, k])
        else:
            arg2 = None
            key2 = f'{T[i, k]}|{W[i, k]}'
            for D, V in zip([D1, D2], [V1, V2]):
                coord = find(D, key2)
                if (arg2 is None) and (coord is not None):
                    arg2 = V[coord]
        assert arg2 is not None
        w = vfunc(cop.get_pair_copula(i, k).hinv2, w, arg2)

    return w
