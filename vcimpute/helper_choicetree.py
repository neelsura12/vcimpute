from vcimpute.helper_diagonalize import is_diagonal_matrix, diagonalize_matrix


class Tree:
    def __init__(self, var):
        self.var = var
        self.children = []
        self.b_only = False

    def add_child(self, other_var):
        child = Tree(other_var)
        self.children.append(child)
        return child

    def __eq__(self, other_var):
        return self.var == other_var

    def __str__(self):
        return str(self.var)


def make_tree(T):
    if not is_diagonal_matrix(T):
        T = diagonalize_matrix(T)
    d = T.shape[0]
    root = Tree(0)
    trees = [root]
    for j in range(d - 1):
        for t in trees:
            if not t.b_only:
                t.add_child(T[d - j - 1, j])
            child = t.add_child(T[d - j - 2, j])
            child.b_only = True
        trees = [child for t in trees for child in t.children]
    return root


def print_tree(node, path=None):
    if len(node.children) == 0:
        print(f'{path}-{node.var}')
    for child in node.children:
        print_tree(child, f'{node.var}' if path is None else f'{path}-{node.var}')


def is_in_tree(node, path):
    if len(path) == 0:
        return True
    for child in node.children:
        if path[0] == child:
            return is_in_tree(child, path[1:] if len(path) > 0 else [])
    return False
