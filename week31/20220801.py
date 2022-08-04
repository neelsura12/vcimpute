import numpy as np
import pyvinecopulib as pv

# %%
s = pv.DVineStructure.simulate(d=5)

d = 5
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
        tmp.append(pv.Bicop(family=pv.BicopFamily.gaussian, parameters=[[0.75]]))
    pc_lst.append(tmp)

cop = pv.Vinecop(s, pc_lst)

U = cop.simulate(n=1000)

ind = {k1: k2 + 1 for k1, k2 in zip(s.order[::-1], range(d))}
nat = {v: k for k, v in ind.items()}

T2 = np.zeros(shape=(d, d), dtype=np.uint64)
for i in range(d):
    for j in range(d - i):
        T2[i, j] = ind[T[i, j]]

Tmax = np.zeros(shape=(d, d), dtype=np.uint64)
for i in range(d):
    for j in range(d - i):
        Tmax[i, j] = np.amax(T2[:(i + 1), j])

# %%

V = np.empty(shape=(d, d), dtype=object)
V2 = np.empty(shape=(d, d), dtype=object)

V[0, d - 1] = U[:, nat[1] - 1]
k = 0
for j in range(d - 1)[::-1]:
    V[0, j] = U[:, nat[d - j] - 1]
    for i in range(0, d - j - 1):
        bicop = cop.get_pair_copula(i, j)
        k = int(d - Tmax[i, j])
        if T2[i, j] == Tmax[i, j]:
            V[i + 1, j] = bicop.hfunc2(np.vstack([V[i, j], V[i, k]]).T)
        else:
            V[i + 1, j] = bicop.hfunc2(np.vstack([V[i, j], V2[i, k]]).T)

        if j > 0:
            if T2[i, j] == Tmax[i, j]:
                V2[i + 1, j] = bicop.hfunc1(np.vstack([V[i, k], V[i, j]]).T)
            else:
                V2[i + 1, j] = bicop.hfunc1(np.vstack(V2[i, k], V[i, j]).T)
