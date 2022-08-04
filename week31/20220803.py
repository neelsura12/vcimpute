import numpy as np
import pyvinecopulib as pv

# %%

d = 3

s = pv.DVineStructure.simulate(d=d)

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

# %%

w = np.random.uniform()
for t in range(1, d - 1)[::-1]:
    col = d - t - 1
    xlab = cop.order[col]
    condset = T[:t, col]
    print(xlab, '|', condset)
    if len(condset) == 1:
        x = U[:, int(condset[0] - 1)][0]
    else:
        x = 0.5

    w = cop.get_pair_copula(d - 2, 0).hinv2([[w, x]])

# %%
