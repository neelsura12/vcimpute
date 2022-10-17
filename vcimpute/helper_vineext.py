import numpy as np
import pyvinecopulib as pv

from vcimpute.simulator import calculate_pseudo_obs
from vcimpute.utils import get, find


def extend_vine(cop_in, U, U_add, family_set, num_threads):
    CC1, CC2, CS, HF1, HF2 = calculate_pseudo_obs(cop_in, U, 0)

    bcop_controls = pv.FitControlsBicop(family_set=family_set, num_threads=num_threads)

    d_in = len(cop_in.order)
    d_out = d_in + 1
    T_in = cop_in.matrix
    T_out = np.zeros(shape=(d_out, d_out), dtype=np.uint64)
    T_out[:-1, 1:] = T_in
    T_out[d_out - 1, 0] = d_out
    avail_vars = sorted(cop_in.order)

    # connect to first tree
    vec1 = U_add
    lst_of_vec = [get(U, i)[:, None] for i in avail_vars]
    idx = get_argmax_kt(vec1, lst_of_vec, family_set)
    T_out[0, 0] = avail_vars[idx]
    del avail_vars[idx]
    bcop = pv.Bicop(data=np.hstack([vec1, lst_of_vec[idx]]), controls=bcop_controls)
    vec1 = bcop.hfunc2(np.hstack([vec1, lst_of_vec[idx]]))[:, None]

    # connect to remaining trees
    for t in range(1, d_out - 1):
        lst_of_vec = []
        eligible_vars = []
        for var in avail_vars:
            cs = ','.join((map(str, sorted(T_out[:t, 0]))))
            key = f'{var}|{cs}'
            for CC, HF in zip([CC1, CC2], [HF1, HF2]):
                coord = find(key, CC)
                if coord is not None:
                    eligible_vars.append(var)
                    lst_of_vec.append(HF[coord])
        lst_of_vec = [vec[:, None] for vec in lst_of_vec]
        idx = get_argmax_kt(vec1, lst_of_vec, family_set)
        T_out[t, 0] = eligible_vars[idx]
        del avail_vars[avail_vars.index(T_out[t, 0])]
        bcop = pv.Bicop(data=np.hstack([vec1, lst_of_vec[idx]]), controls=bcop_controls)
        vec1 = bcop.hfunc2(np.hstack([vec1, lst_of_vec[idx]]))[:, None]

    return T_out


def get_abs_kt(vec1, vec2, family_set):
    bcop_controls = pv.FitControlsBicop(family_set=family_set)
    bcop = pv.Bicop(data=np.hstack([vec1, vec2]), controls=bcop_controls)
    return np.abs(bcop.parameters_to_tau(bcop.parameters))


def get_argmax_kt(vec1, lst_of_vec, family_set):
    max_kt = -1
    max_i = None
    for i, vec2 in enumerate(lst_of_vec):
        kt = get_abs_kt(vec1, vec2, family_set)
        if kt > max_kt:
            max_i = i
            max_kt = kt
    return max_i


def order_miss_vars_by_incr_kendall_tau(miss_vars, rest_vars, U, family_set):
    m = len(rest_vars)
    abs_kts = {}
    for miss_var in miss_vars:
        abs_kts[miss_var] = 0
        for rest_var in rest_vars:
            abs_kts[miss_var] += get_abs_kt(get(U, miss_var)[:, None], get(U, rest_var)[:, None], family_set) / m
    return sorted(abs_kts, key=abs_kts.get)
