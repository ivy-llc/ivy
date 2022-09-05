import ivy


class SparseArray:
    def __init__(self, data=None, *, indices=None, values=None, dense_shape=None):
        if data:
            if indices or values or dense_shape:
                raise ("only specify either data or components")
            self._init(data)
        else:
            pass
        pass

    def _init(self, data):
        if is_ivy_sparse_array(data):
            self._indices = data.indices
            self._values = data.values
            self._dense_shape = data.dense_shape
        else:
            (
                self._indices,
                self._values,
                self._dense_shape,
            ) = self._get_sparse_components(data)

    def _get_sparse_components(self, data):
        pass

    # Properties #
    # -----------#

    @property
    def indices(self):
        return self._indices

    @property
    def values(self):
        return self._values

    @property
    def dense_shape(self):
        return self._dense_shape

    # Setters #
    # --------#

    @indices.setter
    def indices(self, indices):
        pass

    @values.setter
    def values(self, values):
        pass

    @dense_shape.setter
    def dense_shape(self, dense_shape):
        pass


class NativeSparseArray:
    pass


def is_ivy_sparse_array(x):
    return isinstance(x, SparseArray) and is_native_sparse_array(x.data)


def is_native_sparse_array(x):
    return ivy.current_backend(x).is_native_sparse_array(x)
