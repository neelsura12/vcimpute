import numpy as np
import pyvinecopulib as pv

# %%

d = 5

s = pv.RVineStructure.simulate(d=d)

T = np.zeros(shape=(d, d), dtype=np.uint64)
for i in range(d):
    for j in range(d - i - 1):
        T[i, j] = s.struct_array(i, j)
for k, t in zip(range(d), s.order):
    print(d - k - 1)
    T[d - k - 1, k] = t

pc_lst = []
for i in range(d - 1):
    tmp = []
    for j in range(d - i - 1):
        tmp.append(pv.Bicop(family=pv.BicopFamily.gaussian, parameters=[[0.85]]))
    pc_lst.append(tmp)

cop = pv.Vinecop(s, pc_lst)

U = cop.simulate(n=1000)


# %%

def get(X, i):
    return X[:, int(i - 1)]


def vfunc(fun, X1, X2, transpose=True):
    if transpose:
        return fun(np.vstack([np.array(X1), np.array(X2)]).T)
    else:
        return fun(np.vstack([np.array(X1), np.array(X2)]))


def find(D, a_str):
    my_out = np.where(D == a_str)
    if len(my_out) == 2:
        if (len(my_out[0]) == 1) & (len(my_out[1]) == 1):
            return my_out[0][0], my_out[1][0]


# %% forward
V1 = np.empty(shape=(d, d), dtype=object)
V2 = np.empty(shape=(d, d), dtype=object)
W = np.empty(shape=(d, d), dtype=object)
D1 = np.empty(shape=(d, d), dtype=object)
D2 = np.empty(shape=(d, d), dtype=object)

for j in range(d - 1)[::-1]:
    for i in range(d - j - 1):
        W[i, j] = ','.join(list(map(str, sorted(T[:i, j]))))

for j in range(d - 1)[::-1]:
    for i in range(d - j - 1):
        var1 = cop.order[j]
        var2 = T[i, j]
        if W[i, j] == '':
            arg1 = get(U, j)
            arg2 = get(U, cop.order.index(var2))
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
                D1[i, j] = f'{var1}|' + ','.join(sorted(W[i, j].split(',') + [str(var2)]))
                D2[i, j] = f'{var2}|' + ','.join(sorted(W[i, j].split(',') + [str(var1)]))

# %% backward

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

for i in range(1, d):
    true_corr = vfunc(np.corrcoef, get(U, cop.order[0]), get(U, cop.order[i]), False)[0][1]
    est_corr = vfunc(np.corrcoef, w, get(U, cop.order[i]), False)[0][1]
    print(true_corr, est_corr)