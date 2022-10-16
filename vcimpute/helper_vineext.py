import numpy as np
import pyvinecopulib as pv

from vcimpute.utils import make_triangular_array, get, vfunc, find


def extend_vine(cop_in, U, new_to_old_map):
    d = len(cop_in.order) + 1

    T = cop_in.matrix
    HF1 = np.empty(shape=(d, d), dtype=object)
    HF2 = np.empty(shape=(d, d), dtype=object)
    CS = np.empty(shape=(d, d), dtype=object)
    CC1 = np.empty(shape=(d, d), dtype=object)
    CC2 = np.empty(shape=(d, d), dtype=object)
    pair_copulas = make_triangular_array(d-1)

    for j in range(d - 2)[::-1]:
        for i in range(d - j - 2):
            pair_copulas[i][j + 1] = cop_in.get_pair_copula(i, j)
            var1 = cop_in.order[j]
            var2 = T[i, j]
            CS[i][j + 1] = ','.join(list(map(str, sorted(T[:i, j]))))
            if CS[i, j + 1] == '':
                arg1 = get(U, new_to_old_map[var1])
                arg2 = get(U, new_to_old_map[var2])
                HF1[i, j + 1] = vfunc(cop_in.get_pair_copula(i, j).hfunc2, arg1, arg2)
                HF2[i, j + 1] = vfunc(cop_in.get_pair_copula(i, j).hfunc1, arg1, arg2)
                CC1[i, j + 1] = f'{var1}|{var2}'
                CC2[i, j + 1] = f'{var2}|{var1}'
            else:
                arg1, arg2 = None, None
                key1 = f'{var1}|{CS[i, j + 1]}'
                key2 = f'{var2}|{CS[i, j + 1]}'
                for CC, HF in zip([CC1, CC2], [HF1, HF2]):
                    coord = find(CC, key1)
                    if (arg1 is None) and (coord is not None):
                        arg1 = HF[coord]
                    coord = find(CC, key2)
                    if (arg2 is None) and (coord is not None):
                        arg2 = HF[coord]
                assert (arg1 is not None) and (arg2 is not None)
                HF1[i, j + 1] = vfunc(cop_in.get_pair_copula(i, j).hfunc2, arg1, arg2)
                HF2[i, j + 1] = vfunc(cop_in.get_pair_copula(i, j).hfunc1, arg1, arg2)
                CC1[i, j + 1] = f'{var1}|' + ','.join(sorted(CS[i, j + 1].split(',') + [str(var2)]))
                CC2[i, j + 1] = f'{var2}|' + ','.join(sorted(CS[i, j + 1].split(',') + [str(var1)]))

    bcop_controls = pv.FitControlsBicop(family_set=[pv.BicopFamily.gaussian]) # generify

    T_new = np.zeros(shape=(d, d), dtype=np.uint64)
    T_new[d - 1, 0] = d
    T_new[:-1, 1:] = cop_in.matrix

    j = 0
    for i in range(d - 1):
        var1 = d
        var2 = cop_in.order[d - i - 2]
        T_new[i, j] = var2
        CS[i, j] = ','.join(list(map(str, sorted(T_new[:i, j]))))
        if CS[i, j] == '':
            arg1 = get(U, new_to_old_map[var1])
            arg2 = get(U, new_to_old_map[var2])
            bcop = pv.Bicop(data=np.vstack([arg1, arg2]).T, controls=bcop_controls)
            pair_copulas[i][j] = bcop
            HF1[i, j] = vfunc(bcop.hfunc2, arg1, arg2)
            HF2[i, j] = vfunc(bcop.hfunc1, arg1, arg2)
            CC1[i, j] = f'{var1}|{var2}'
            CC2[i, j] = f'{var2}|{var1}'
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
            assert (arg1 is not None) and (arg2 is not None)
            bcop = pv.Bicop(data=np.vstack([arg1, arg2]).T, controls=bcop_controls)
            pair_copulas[i][j] = bcop
            HF1[i, j] = vfunc(bcop.hfunc2, arg1, arg2)
            HF2[i, j] = vfunc(bcop.hfunc1, arg1, arg2)
            CC1[i, j] = f'{var1}|' + ','.join(sorted(CS[i, j].split(',') + [str(var2)]))
            CC2[i, j] = f'{var2}|' + ','.join(sorted(CS[i, j].split(',') + [str(var1)]))

    cop_out = pv.Vinecop(T_new, pair_copulas)
    return cop_out
