import ivy


class SparseArray:
    def __init__(self, data=None, *, indices=None, values=None, dense_shape=None):
        if data:
            if indices or values or dense_shape:
                raise Exception("only specify either data or components")
            self._init_data()
        else:
            if indices and values and dense_shape:
                self._init_coo_components(indices, values, dense_shape)
            # TODO: to add csr
            else:
                raise Exception("indices, values and dense_shape must all be specified")

    def _init_data(self, data):
        if is_ivy_sparse_array(data):
            self._data = data.data
            self._indices = data.indices
            self._values = data.values
            self._dense_shape = data.dense_shape
        else:
            assert ivy.is_native_sparse_array(data), "not a native sparse array"
            self._data = data
            self._init_native_components()

    def _init_native_components(self):
        indices, values, shape = ivy.current_backend().init_native_components(
            self._data
        )
        self._indices = ivy.array(indices, dtype="int64")
        self._values = ivy.array(values, dtype=values.dtype)
        self._dense_shape = ivy.Shape(shape)

    def _init_coo_components(self, indices, values, shape):
        indices = ivy.array(indices, dtype="int64")
        values = ivy.array(values)
        shape = ivy.Shape(shape)
        self._verify_coo_components(indices=indices, values=values, dense_shape=shape)
        self._indices = indices
        self._values = values
        self._dense_shape = shape
        self._data = ivy.current_backend().init_data_sparse_array(
            indices, values, shape
        )

    def _verify_coo_components(self, *, indices=None, values=None, dense_shape=None):
        indices = indices if indices else self._indices
        values = values if values else self._values
        dense_shape = dense_shape if dense_shape else self._dense_shape
        # coordinates style, must be shaped (x, y)
        assert len(indices.shape) == 2, "indices must be 2D"
        assert (
            len(dense_shape) == indices.shape[0]
        ), "shape and indices shape do not match"
        # number of values must match number of coordinates
        assert values.shape[0] == indices.shape[1], "values and indices do not match"
        for i in range(indices.shape[0]):
            assert ivy.all(
                ivy.less(indices[i], dense_shape[i])
            ), "indices is larger than shape"

    # Properties #
    # -----------#

    @property
    def data(self):
        return self._data

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

    @data.setter
    def data(self, data):  # TODO
        assert ivy.is_native_sparse_array(data)
        self._init_data(data)

    @indices.setter
    def indices(self, indices):
        indices = ivy.array(indices, dtype="int64")
        self._verify_coo_components(indices=indices)
        self._indices = indices

    @values.setter
    def values(self, values):
        values = ivy.array(values)
        self._verify_coo_components(values=values)
        self._values = values

    @dense_shape.setter
    def dense_shape(self, dense_shape):
        dense_shape = ivy.Shape(dense_shape)
        self._verify_coo_components(dense_shape=dense_shape)
        self._dense_shape = dense_shape

    # Instance Methods #
    # ---------------- #

    def to_dense_array(self, *, native=False):
        new_ind = []
        for i in range(self._values.shape[0]):
            coordinate = ivy.gather(self._indices, ivy.array([[i]]))
            coordinate = ivy.reshape(coordinate, (self._indices.shape[0],))
            new_ind.append(coordinate.to_list())
        ret = ivy.scatter_nd(
            ivy.array(new_ind), self._values, ivy.array(self._dense_shape)
        )
        return ret.to_native() if native else ret


class NativeSparseArray:
    pass


def is_ivy_sparse_array(x):
    return isinstance(x, SparseArray)


def is_native_sparse_array(x):  # TODO
    return ivy.current_backend(x).is_native_sparse_array(x)
