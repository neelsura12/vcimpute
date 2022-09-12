import numpy as np
import pyvinecopulib as pv

bicop_family_map = {
    'gaussian': pv.BicopFamily.gaussian,
    'student': pv.BicopFamily.student,
    'clayton': pv.BicopFamily.clayton,
    'frank': pv.BicopFamily.frank
}


def make_triangular_array(d):
    pair_copulas = [None for _ in range(d - 1)]
    for j in range(d - 1)[::-1]:
        pair_copulas[j] = [None for _ in range(d - j - 1)]
    return pair_copulas


def make_copula(d):
    structure = pv.RVineStructure.simulate(d)

    pair_copulas = []
    for j in range(d - 1):
        tmp = []
        pair_copulas.append(tmp)
        for _ in range(d - j - 1):
            rho = np.minimum(np.maximum(np.random.beta(1, 0.75), 0.01), 0.99)
            tmp.append(pv.Bicop(family=pv.BicopFamily.gaussian, parameters=[[rho]]))

    cop = pv.Vinecop(structure, pair_copulas)

    return cop


def get_order(T):
    d = T.shape[0]
    return [T[d - j - 1, j] for j in range(d)]


def get_pair_copulas(cop):
    T = cop.matrix
    d = T.shape[0]
    pair_copulas = make_triangular_array(d)
    for j in range(d):
        for i in range(d - j - 1):
            pair_copulas[i][j] = cop.get_pair_copula(i, j)
    return pair_copulas


def get(X, var):
    return X[:, int(var - 1)]


def is_leaf_in_all_subtrees(T, var):
    d = T.shape[0]
    return (T[d - 2, 0] == var) or (T[d - 1, 0] == var)


def vfunc(fun, X1, X2, transpose=True):
    if transpose:
        return fun(np.vstack([np.array(X1), np.array(X2)]).T)
    else:
        return fun(np.vstack([np.array(X1), np.array(X2)]))


def find(D, a_str):
    coord = np.argwhere(D == a_str)
    if coord.shape[0] == 1:
        return tuple(coord[0])
