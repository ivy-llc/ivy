# global
import ivy


class Param:

    def __init__(self, ptype, tree_depth, is_var=False, shape=None):
        self._count = 0
        self._ptype = ptype
        self._tree_depth = tree_depth
        self._param_stack = list()
        self._is_var = is_var
        self._shape = tuple(shape) if ivy.exists(shape) else None

    def set(self, val):
        self._param_stack = [val]*self._count

    def set_count(self, count):
        self._count = count

    def get(self):
        return self._param_stack.pop()

    def __repr__(self):
        return '<Param, type={}, depth={}, count={}, current={}>'.format(
            self._ptype, self._tree_depth, self._count, len(self._param_stack))

    def __len__(self):
        return len(self._param_stack)

    @property
    def count(self):
        return self._count

    @property
    def depth(self):
        return self._tree_depth

    @property
    def ptype(self):
        return self._ptype

    @property
    def is_var(self):
        return self._is_var

    @property
    def shape(self):
        return self._shape
