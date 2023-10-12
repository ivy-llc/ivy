def tree_leaves(tree, is_leaf=None):
    # todo: is_leaf
    if isinstance(tree, (tuple, list)):
        new_struc = []
        for child in tree:
            new_struc += tree_leaves(child)
        return new_struc
    elif isinstance(tree, dict):
        new_struc = []
        for key in sorted(tree):
            new_struc += tree_leaves(tree[key])
        return new_struc
    return [tree]


def tree_map(f, tree, *rest, is_leaf=None):
    # todo: is_leaf
    is_tuple = isinstance(tree, tuple)
    if is_tuple:
        tree = list(tree)

    if isinstance(tree, list):
        for idx, elem in enumerate(tree):
            curr_r = [r[idx] for r in rest] if rest else []
            tree[idx] = tree_map(f, tree[idx], *curr_r, is_leaf=is_leaf)
        return tuple(tree) if is_tuple else tree
    elif isinstance(tree, dict):
        for key in sorted(tree):
            curr_r = [r[key] for r in rest] if rest else []
            tree[key] = tree_map(f, tree[key], *curr_r, is_leaf=is_leaf)
        return tree
    return f(tree, *rest) if rest else f(tree)
