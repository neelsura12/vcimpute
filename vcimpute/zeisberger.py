import numpy as np
import pyvinecopulib as pv

from vcimpute.constants import bicop_family_map
from vcimpute.helper_diagonalize import diagonalize_copula
from vcimpute.helper_subvines import find_subvine_structures, remove_var
from vcimpute.helper_vinestructs import generate_r_vine_structure, generate_c_or_d_vine_structure
from vcimpute.helper_vinestructs import relabel_vine_mat
from vcimpute.simulator import simulate_order_k
from vcimpute.utils import get
from vcimpute.utils import make_triangular_array, is_leaf_in_all_subtrees


class VineCopReg:
    def __init__(self, bicop_families, num_threads, vine_structure):
        family_set = [bicop_family_map[k] for k in bicop_families]
        self.controls = pv.FitControlsVinecop(family_set=family_set, num_threads=num_threads)
        assert vine_structure in ['R', 'C', 'D']
        self.vine_structure = vine_structure

    def fit_transform(self, X_mis):
        d = X_mis.shape[1]
        n_mis = np.sum(np.any(np.isnan(X_mis), axis=0))

        # check monotonic
        for j in range(d):
            this_var_nan = np.isnan(X_mis[:, j])
            assert np.all(np.isnan(X_mis[this_var_nan, (j + 1):])), 'non-monotonic missingness pattern'

        # simulate vine structure for sequential imputation
        structure = None
        if self.vine_structure == 'R':
            structure = generate_r_vine_structure(d, n_mis)
        elif self.vine_structure == 'C':
            structure = generate_c_or_d_vine_structure(d, n_mis, pv.CVineStructure)
        elif self.vine_structure == 'D':
            structure = generate_c_or_d_vine_structure(d, n_mis, pv.DVineStructure)
        assert structure is not None

        # make copula with fixed structure
        pcs = make_triangular_array(d)
        for j in range(d - 1):
            for i in range(d - j - 1):
                pcs[i][j] = pv.Bicop()
        cop = pv.Vinecop(structure=structure, pair_copulas=pcs)

        X_imp = np.copy(X_mis)
        for k in range(n_mis)[::-1]:
            cop.select(X_imp, controls=self.controls)
            x_imp = simulate_order_k(cop, X_imp, k)
            assert not np.any(np.isnan(x_imp)), 'check imputation order'

            x_mis = get(X_imp, cop.order[k])
            is_missing = np.isnan(x_mis)
            x_mis[is_missing] = x_imp[is_missing]

        assert not np.any(np.isnan(X_imp)), 'invalid state, not all values imputed'
        return X_imp


class VineCopFit:
    def __init__(self, bicop_families, num_threads):
        family_set = [bicop_family_map[k] for k in bicop_families]
        self.controls = pv.FitControlsVinecop(family_set=family_set, num_threads=num_threads)

    def fit_transform(self, X_mis):
        # fit on complete cases
        cop_orig = pv.Vinecop(data=X_mis, controls=self.controls)
        T_orig = cop_orig.matrix
        pcs_orig = cop_orig.pair_copulas
        d_orig = T_orig.shape[0]

        # order from least missing to most
        miss_vars, = np.where(np.count_nonzero(np.isnan(X_mis), axis=0))
        miss_vars += 1
        miss_vars = list(miss_vars.astype(np.uint64))

        X_imp = np.copy(X_mis)
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
                T_sub_relabel = relabel_vine_mat(T_sub, old_to_new)
                cop_sub = pv.Vinecop(structure=pv.RVineStructure(T_sub_relabel), pair_copulas=pcs_sub)
                X_imp_sub = X_imp[:, [int(new_to_old[i + 1] - 1) for i in range(len(new_to_old))]]

                if T_sub[d_sub - 2, 0] == cur_var_mis:
                    cop_sub_diag = diagonalize_copula(cop_sub, old_to_new[cur_var_mis])
                    ximp_lst.append(simulate_order_k(cop_sub_diag, X_imp_sub, 0))
                    imputed = True

                if T_sub[d_sub - 1, 0] == cur_var_mis:
                    ximp_lst.append(simulate_order_k(cop_sub, X_imp_sub, 0))
                    imputed = True

                # only keep the last imputation since it uses all available information
                if imputed and (d_sub == d_orig):
                    ximp_lst = ximp_lst[-1]
                    break

            # average imputations
            ximp_mat = np.vstack(ximp_lst).T
            n_avail = ximp_mat.shape[1] - np.count_nonzero(np.isnan(ximp_mat), axis=1)
            assert np.all(n_avail) > 0

            # insert imputed values back
            idx_mis = int(cur_var_mis - 1)
            missing = np.isnan(X_imp[:, idx_mis])
            if np.any(missing):
                ximp = np.nansum(ximp_mat, axis=1) / n_avail
                X_imp[missing, idx_mis] = ximp[missing]
        return X_imp
