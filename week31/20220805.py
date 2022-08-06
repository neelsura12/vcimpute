import numpy as np
import pyvinecopulib as pv

from vcimpute.simulator import vfunc, get, simulate_order0

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
        tmp.append(pv.Bicop(family=pv.BicopFamily.gaussian, parameters=[[0.5]]))
    pc_lst.append(tmp)

cop = pv.Vinecop(s, pc_lst)

U = cop.simulate(n=1000)

#%%
w = simulate_order0(cop, U)

for i in range(1, d):
    true_corr = vfunc(np.corrcoef, get(U, cop.order[0]), get(U, cop.order[i]), False)[0][1]
    est_corr = vfunc(np.corrcoef, w, get(U, cop.order[i]), False)[0][1]
    print(true_corr, est_corr)
