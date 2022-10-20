import numpy as np
import pyvinecopulib as pv

from vcimpute.helper_choicetree import make_tree, is_in_tree
from vcimpute.helper_diagonalize import diagonalize_matrix
from vcimpute.helper_mdp import all_miss_vars, mdp_coords, old_to_new, sort_miss_vars_by_increasing_miss_vars
from vcimpute.helper_vineext import extend_vine, order_miss_vars_by_incr_kendall_tau
from vcimpute.helper_vinestructs import relabel_vine_matrix, generate_r_vine_structure
from vcimpute.utils import bicop_family_map, get_order


class MdpFit:

    def __init__(self, bicop_family, num_threads, seed):
        self.bicop_family = bicop_family_map[bicop_family]
        self.num_threads = num_threads
        self.controls = pv.FitControlsVinecop(family_set=[self.bicop_family], num_threads=self.num_threads)
        self.X_imp = None
        self.cop = None
        self.d = None

    def fit_transform(self, X_mis):
        self.d = X_mis.shape[1]
        self.X_imp = np.copy(X_mis)
        all_vars = 1 + np.arange(self.d, dtype='uint64')
        family_set = [self.bicop_family]
        self.cop = pv.Vinecop(d=self.d)
        self.cop.select(self.X_imp, self.controls)
        while np.any(np.isnan(self.X_imp)):
            non_adhoc_patterns = self.impute_adhoc()
            if not np.any(np.isnan(self.X_imp)):
                break
            non_adhoc_patterns = sort_miss_vars_by_increasing_miss_vars(non_adhoc_patterns)
            miss_vars = non_adhoc_patterns[0]
            rest_vars = np.setdiff1d(all_vars, miss_vars)
            miss_vars = order_miss_vars_by_incr_kendall_tau(miss_vars, rest_vars, self.X_imp, family_set)
            if len(miss_vars) < self.d - 1:
                cop_in = pv.Vinecop(d=len(rest_vars))
                U = self.X_imp[:, rest_vars - 1]
                cop_in.select(U, self.controls)
                old_to_new = {k: (i + 1) for i, k in enumerate(rest_vars)}
                T_out = None
                for var in miss_vars:
                    U_add = self.X_imp[:, [int(var - 1)]]
                    T_out = extend_vine(cop_in, U, U_add, family_set, self.num_threads)
                    cop_in = pv.Vinecop(structure=pv.RVineStructure(T_out))
                    U = np.hstack([U, U_add])
                    cop_in.select(data=U, controls=self.controls)
                    old_to_new[var] = U.shape[1]
                new_to_old = {v: k for k, v in old_to_new.items()}
                T = relabel_vine_matrix(T_out, new_to_old)
                structure = pv.RVineStructure(T)
            else:
                structure = generate_r_vine_structure(miss_vars, rest_vars)
            self.cop = pv.Vinecop(structure=structure)
            self.cop.select(self.X_imp, self.controls)
            self.impute(miss_vars)
            if not np.any(np.isnan(self.X_imp)):
                break

        return self.X_imp

    def impute_adhoc(self):
        root = make_tree(self.cop.matrix)
        mdp_vars = all_miss_vars(self.X_imp)
        mdp_vars_ordered = get_ordered_miss_vars(mdp_vars, get_order(diagonalize_matrix(self.cop.matrix)))

        non_adhoc_patterns = []
        for miss_vars in mdp_vars_ordered:
            if is_in_tree(root, miss_vars):
                self.impute(miss_vars)
            else:
                non_adhoc_patterns.append(miss_vars)
        return non_adhoc_patterns

    def impute(self, miss_vars):
        miss_vars = np.array(miss_vars, dtype='uint64')
        miss_idx = miss_vars - 1
        mdp = np.zeros(shape=(self.d,), dtype='bool')
        mdp[miss_idx] = True
        miss_rows = mdp_coords(self.X_imp, mdp)

        rb = self.cop.rosenblatt(self.X_imp[miss_rows])
        rb[np.isnan(rb)] = np.random.uniform(size=np.count_nonzero(np.isnan(rb)))
        irb = self.cop.inverse_rosenblatt(rb)
        for i in range(len(miss_rows)):
            self.X_imp[miss_rows[i], miss_idx] = irb[i, miss_idx]


def get_ordered_miss_vars(mdp_vars, order):
    d = len(order)
    old_to_new_dct = old_to_new(order, 1 + np.arange(d))
    new_to_old_dct = {v: k for k, v in old_to_new_dct.items()}

    mdp_indices_ordered = []
    for i in range(mdp_vars.shape[0]):
        mdp_idx = mdp_vars[i]
        mdp_idx = mdp_idx[mdp_idx != 0]
        mdp_idx_ordered = sorted(map(lambda x: old_to_new_dct[x], mdp_idx))
        mdp_idx = list(map(lambda x: new_to_old_dct[x], mdp_idx_ordered))
        mdp_indices_ordered.append(mdp_idx)
    return mdp_indices_ordered
