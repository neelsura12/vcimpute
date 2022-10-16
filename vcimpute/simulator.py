import numpy as np

from vcimpute.utils import get, vfunc, find


def simulate_order_k(cop, U, k):
    d = U.shape[1]
    T = cop.matrix

    # forward
    CC1, CC2, CS, HF1, HF2 = calculate_pseudo_obs(cop, U, k)

    # backward
    w = np.random.uniform(size=U.shape[0])
    for i in range(d - k - 1)[::-1]:
        if CS[i, k] == '':
            arg2 = get(U, T[i, k])
        else:
            arg2 = None
            key2 = f'{T[i, k]}|{CS[i, k]}'
            for CC, HF in zip([CC1, CC2], [HF1, HF2]):
                coord = find(CC, key2)
                if (arg2 is None) and (coord is not None):
                    arg2 = HF[coord]
        assert arg2 is not None
        w = vfunc(cop.get_pair_copula(i, k).hinv2, w, arg2)

    return w


def calculate_pseudo_obs(cop, U, k):
    d = U.shape[1]
    T = cop.matrix

    HF1 = np.empty(shape=(d, d), dtype=object)
    HF2 = np.empty(shape=(d, d), dtype=object)
    CS = np.empty(shape=(d, d), dtype=object)
    CC1 = np.empty(shape=(d, d), dtype=object)
    CC2 = np.empty(shape=(d, d), dtype=object)

    for j in range(k, d - 1)[::-1]:
        for i in range(d - j - 1):
            var1 = cop.order[j]
            var2 = T[i, j]
            CS[i, j] = ','.join(list(map(str, sorted(T[:i, j]))))
            if CS[i, j] == '':
                arg1 = get(U, var1)
                arg2 = get(U, var2)
                HF1[i, j] = vfunc(cop.get_pair_copula(i, j).hfunc1, arg1, arg2)
                HF2[i, j] = vfunc(cop.get_pair_copula(i, j).hfunc2, arg1, arg2)
                CC1[i, j] = f'{var2}|{var1}'
                CC2[i, j] = f'{var1}|{var2}'
            else:
                arg1, arg2 = None, None
                key1 = f'{var1}|{CS[i, j]}'
                key2 = f'{var2}|{CS[i, j]}'
                for CC, HF in zip([CC1, CC2], [HF1, HF2]):
                    coord = find(CC, key1)
                    if (arg1 is None) and (coord is not None):
                        arg1 = HF[coord]
                    coord = find(CC, key2)
                    if (arg2 is None) and (coord is not None):
                        arg2 = HF[coord]
                if (arg1 is not None) and (arg2 is not None):
                    HF1[i, j] = vfunc(cop.get_pair_copula(i, j).hfunc1, arg1, arg2)
                    HF2[i, j] = vfunc(cop.get_pair_copula(i, j).hfunc2, arg1, arg2)
                    CC1[i, j] = f'{var2}|' + ','.join(map(str, sorted(list(map(int, CS[i, j].split(','))) + [var1])))
                    CC2[i, j] = f'{var1}|' + ','.join(map(str, sorted(list(map(int, CS[i, j].split(','))) + [var2])))

    return CC1, CC2, CS, HF1, HF2
