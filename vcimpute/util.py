import numpy as np
import pyvinecopulib as pv


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


def make_triangular_array(d):
    pair_copulas = np.empty(shape=(d - 1,), dtype='object')
    for j in range(d - 1)[::-1]:
        pair_copulas[j] = list(np.empty(shape=(d - j - 1,), dtype='object'))
    return list(pair_copulas)


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


def vine_structure_to_matrix(structure):
    d = len(structure.order)
    mat = np.zeros(shape=(d, d), dtype=np.uint64)
    for t in range(d - 1):
        for e in range(d - t - 1):
            mat[t, e] = structure.struct_array(t, e)
    for j in range(d):
        mat[d - j - 1, j] = structure.order[j]
    return mat
