import numpy as np
import pandas as pd
import pyvinecopulib as pv
from gcimpute.gaussian_copula import GaussianCopula
from gcimpute.helper_evaluation import get_smae
from joblib import Parallel, delayed
from matplotlib import pyplot as plt

from vcimpute.simulator import get, simulate_order0


# test vc impute vs. gc impute using true model

# %%

def compare(n, d, p, R):
    out = []
    for _ in range(R):
        structure = pv.CVineStructure.simulate(d)

        pair_copulas = []
        for j in range(d - 1):
            tmp = []
            pair_copulas.append(tmp)
            for _ in range(d - j - 1):
                rho = np.minimum(np.maximum(np.random.beta(1, 0.75), 0.01), 0.99)
                tmp.append(pv.Bicop(family=pv.BicopFamily.gaussian, parameters=[[rho]]))

        cop = pv.Vinecop(structure, pair_copulas)

        U = cop.simulate(n=n, seeds=list(1 + np.arange(d)))
        v_idx = cop.order[0]

        missing = np.random.binomial(n=1, p=p, size=U.shape[0])

        U_mask = np.copy(U)
        v_mask = get(U_mask, v_idx)
        v_mask[missing == 1] = np.nan

        # vcimpute
        w = simulate_order0(cop, U_mask)
        U_imp1 = np.copy(U_mask)
        v_imp = get(U_imp1, v_idx)
        v_imp[missing == 1] = w[missing == 1]

        # gcimpute
        model = GaussianCopula()
        U_imp2 = model.fit_transform(X=U_mask)

        smae1 = get_smae(U_imp1, U, U_mask)[int(v_idx - 1)]
        re1 = (np.linalg.norm(np.corrcoef(U_imp1.T) - np.corrcoef(U.T), ord='fro') /
               np.linalg.norm(np.corrcoef(U.T), ord='fro'))

        smae2 = get_smae(U_imp2, U, U_mask)[int(v_idx - 1)]
        re2 = (np.linalg.norm(np.corrcoef(U_imp2.T) - np.corrcoef(U.T), ord='fro') /
               np.linalg.norm(np.corrcoef(U.T), ord='fro'))

        out.append((smae1, smae2, re1, re2))

    df = pd.DataFrame(out, columns=['smae1', 'smae2', 're1', 're2']).assign(n=n, d=d, p=p, R=R)
    return df


# %%

out = Parallel(n_jobs=-1)(delayed(compare)(1000, d, p, 100) for d in range(8,25) for p in np.arange(0.01, 0.1, 0.01))

# %%
df = pd.concat(out)
df.to_pickle('res_true_model_test2.pkl')

# %%
df2 = df.groupby(['d', 'p'])[['smae1', 'smae2']].mean().reset_index('p')
for d in df2.index.unique():
    df3 = df2.loc[d]
    plt.figure()
    plt.plot(df3['p'], df3['smae1'], label='vc')
    plt.plot(df3['p'], df3['smae2'], label='gc')
    plt.title(d)
    plt.legend()
    plt.show()

# %%


plt.scatter(df['smae1'], df['smae2'])
ax = plt.gca()
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),
    np.max([ax.get_xlim(), ax.get_ylim()]),
]
ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)
plt.xlabel('smae1 (vc true)')
plt.ylabel('smae2 (gc)')
plt.show()
