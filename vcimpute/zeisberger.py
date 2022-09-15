
import numpy as np
import pyvinecopulib as pv

from vcimpute.helper_diagonalize import diagonalize_copula
from vcimpute.helper_mdp import all_mdps, sort_mdps_by_increasing_missing_vars, mdp_coords, count_missing_by_col
from vcimpute.helper_subvines import find_subvine_structures, remove_var
from vcimpute.helper_vinestructs import generate_r_vine_structure, relabel_vine_matrix
from vcimpute.simulator import simulate_order_k
from vcimpute.utils import get, bicop_family_map, make_triangular_array, is_leaf_in_all_subtrees



class VineCopReg:
    def __init__(self, bicop_family, num_threads, vine_structure, is_monotone, seed):
        self.bicop_family = bicop_family_map[bicop_family]
        self.controls = pv.FitControlsVinecop(family_set=[self.bicop_family], num_threads=num_threads)
        assert vine_structure in ['R', 'C', 'D']
        self.vine_structure = vine_structure
        self.is_monotone = is_monotone
        self.rng = np.random.default_rng(seed)

    def fit_transform(self, X_mis):
        X_imp = np.copy(X_mis)
        if self.is_monotone:
            self.impute(X_imp, [])
        else:
            mdps = sort_mdps_by_increasing_missing_vars(all_mdps(X_imp))
            for mdp in mdps:
                self.impute(X_imp, mdp)

        assert not np.any(np.isnan(X_imp)), 'invalid state, not all values imputed'
        return X_imp

    def impute(self, X_imp, mdp):
        d = X_imp.shape[1]

        if self.is_monotone:
            n_miss_by_col = count_missing_by_col(X_imp)
            miss_vars = 1 + np.where(n_miss_by_col)[0]
            miss_vars = list(miss_vars[n_miss_by_col[np.where(n_miss_by_col)[0]].argsort()])[::-1]
            miss_coords = np.where(np.any(np.isnan(X_imp), axis=1))[0]
        else:
            miss_vars = list(1 + np.where(mdp)[0])
            self.rng.shuffle(miss_vars)
            miss_coords = mdp_coords(X_imp, mdp)
        obs_vars = list(set(1 + np.arange(d)).difference(miss_vars))
        self.rng.shuffle(obs_vars)

        # simulate vine structure for sequential imputation
        structure = None
        if self.vine_structure == 'R':
            structure = generate_r_vine_structure(miss_vars, obs_vars)
        elif self.vine_structure == 'C':
            structure = pv.CVineStructure(order=miss_vars + obs_vars)
        elif self.vine_structure == 'D':
            structure = pv.DVineStructure(order=miss_vars + obs_vars)
        assert structure is not None

        # make copula with fixed structure
        pcs = make_triangular_array(d)
        for j in range(d - 1):
            for i in range(d - j - 1):
                pcs[i][j] = pv.Bicop()
        cop = pv.Vinecop(structure=structure, pair_copulas=pcs)

        # fit to complete cases
        for k in range(len(miss_vars))[::-1]:
            var_mis = miss_vars[k]
            cop.select(X_imp, controls=self.controls)
            assert cop.order[k] == var_mis
            x_imp = simulate_order_k(cop, X_imp, k)
            x_mis = get(X_imp, var_mis)
            assert not np.any(np.isnan(x_imp[miss_coords])), 'check imputation order'
            x_mis[miss_coords] = x_imp[miss_coords]


class VineCopFit:
    def __init__(self, bicop_family, num_threads, is_monotone, seed):
        self.bicop_family = bicop_family_map[bicop_family]
        self.controls = pv.FitControlsVinecop(family_set=[self.bicop_family], num_threads=num_threads)
        self.is_monotone = is_monotone
        self.rng = np.random.default_rng(seed)

    def fit_transform(self, X_mis):
        # fit on complete cases
        cop_orig = pv.Vinecop(data=X_mis, controls=self.controls)
        T_orig = cop_orig.matrix
        pcs_orig = cop_orig.pair_copulas
        d_orig = T_orig.shape[0]

        X_imp = np.copy(X_mis)
        if self.is_monotone:
            self.impute(X_imp, [], T_orig, pcs_orig, d_orig)
        else:
            mdps = all_mdps(X_imp)
            for mdp in mdps:
                self.impute(X_imp, mdp, T_orig, pcs_orig, d_orig)

        assert not np.any(np.isnan(X_imp)), 'invalid state, not all values imputed'
        return X_imp

    def impute(self, X_imp, mdp, T_orig, pcs_orig, d_orig):

        if self.is_monotone:
            n_miss_by_col = count_missing_by_col(X_imp)
            miss_vars = 1 + np.where(n_miss_by_col)[0]
            miss_vars = list(miss_vars[n_miss_by_col[np.where(n_miss_by_col)[0]].argsort()])
            miss_coords = np.where(np.any(np.isnan(X_imp), axis=1))[0]
        else:
            miss_vars = list(1 + np.where(mdp)[0])
            self.rng.shuffle(miss_vars)
            miss_coords = mdp_coords(X_imp, mdp)

        for cur_var_mis in miss_vars:
            # remove as-yet missing values
            T, pcs = T_orig, pcs_orig
            for rest_var_mis in miss_vars[(miss_vars.index(cur_var_mis) + 1):]:
                T, pcs = remove_var(T, pcs, rest_var_mis)
            subvine_structures = find_subvine_structures(T, pcs, cur_var_mis)

            # collect cur_var_mis imputed values per sub-vine structure
            ximp_lst = []
            for T_sub, pcs_sub in subvine_structures:
                imputed = False
                d_sub = T_sub.shape[0]
                assert is_leaf_in_all_subtrees(T_sub, cur_var_mis)

                # relabel indices
                ordered_old_vars = filter(lambda x: x != 0, np.unique(T_sub))
                old_to_new = {var_old: k + 1 for k, var_old in enumerate(ordered_old_vars)}
                new_to_old = {v: k for k, v in old_to_new.items()}
                T_sub_relabel = relabel_vine_matrix(T_sub, old_to_new)
                cop_sub = pv.Vinecop(structure=pv.RVineStructure(T_sub_relabel), pair_copulas=pcs_sub)
                X_imp_sub = X_imp[:, [int(new_to_old[i + 1] - 1) for i in range(len(new_to_old))]]

                if T_sub[d_sub - 2, 0] == cur_var_mis:
                    cop_sub_diag = diagonalize_copula(cop_sub, old_to_new[cur_var_mis], self.bicop_family)
                    ximp_lst.append(simulate_order_k(cop_sub_diag, X_imp_sub, 0))
                    imputed = True

                if T_sub[d_sub - 1, 0] == cur_var_mis:
                    ximp_lst.append(simulate_order_k(cop_sub, X_imp_sub, 0))
                    imputed = True

                # only keep the last imputation since it uses all available information
                if imputed and (d_sub == d_orig):
                    ximp_lst = [ximp_lst[-1]]
                    break

            # average imputations
            ximp_mat = np.vstack(ximp_lst).T
            ximp_mat = ximp_mat[miss_coords]
            n_avail = ximp_mat.shape[1] - np.count_nonzero(np.isnan(ximp_mat), axis=1)
            assert np.all(n_avail) > 0

            # insert imputed values back
            idx_mis = int(cur_var_mis - 1)
            ximp = np.nansum(ximp_mat, axis=1) / n_avail
            X_imp[miss_coords, idx_mis] = ximp

        return X_imp
