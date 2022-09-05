import ivy


class SparseArray:
    def __init__(self, data=None, *, indices=None, values=None, dense_shape=None):
        if data:
            if indices or values or dense_shape:
                raise Exception("only specify either data or components")
            self._init_data()  # TODO
        else:
            self._init_components(indices, values, dense_shape)
        pass  # TODO

    def _init_data(self, data):  # TODO
        if is_ivy_sparse_array(data):
            self._data = data.data
        else:
            assert ivy.is_native_sparse_array(data)
            self._data = data
        self._get_sparse_components()

    def _init_components(self, indices, values, shape):
        indices = ivy.array(indices, dtype="int64")
        values = ivy.array(values)
        shape = ivy.Shape(shape)
        # coordinates style, must be shaped (x, y)
        assert len(indices.shape) == 2, "indices must be 2D"
        assert len(shape) == indices.shape[0], "shape and indices shape do not match"
        # number of values must match number of coordinates
        assert values.shape[0] == indices.shape[1], "values and indices do not match"
        for i in range(indices.shape[0]):
            assert ivy.all(
                ivy.less(indices[i], shape[i])
            ), "indices is larger than shape"
        self._indices = indices
        self._values = values
        self._dense_shape = shape
        # TODO: data

    def _get_sparse_components(self):  # T TODO
        indices, values, shape = ivy.current_backend(self._data).get_sparse_components(
            self._data
        )
        self._indices = ivy.array(indices, dtype="int64")
        self._values = ivy.array(values, dtype=values.dtype)
        self._dense_shape = ivy.Shape(shape)

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
        assert len(indices.shape) == 2, "indices must be 2D"
        assert (
            len(self._dense_shape) == indices.shape[0]
        ), "shape and indices shape do not match"
        # number of values must match number of coordinates
        assert (
            self._values.shape[0] == indices.shape[1]
        ), "values and indices do not match"
        for i in range(indices.shape[0]):
            assert ivy.all(
                ivy.less(indices[i], self._shape[i])
            ), "indices is larger than shape"
        self._indices = indices

    @values.setter
    def values(self, values):
        values = ivy.array(values)
        assert (
            values.shape[0] == self._indices.shape[1]
        ), "values and indices do not match"
        self._values = values

    @dense_shape.setter
    def dense_shape(self, dense_shape):
        dense_shape = ivy.Shape(dense_shape)
        assert (
            len(dense_shape) == self._indices.shape[0]
        ), "shape and indices shape do not match"
        for i in range(self._indices.shape[0]):
            assert ivy.all(
                ivy.less(self._indices[i], dense_shape[i])
            ), "indices is larger than shape"
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


def is_ivy_sparse_array(x):  # TODO
    return isinstance(x, SparseArray) and is_native_sparse_array(x.data)


def is_native_sparse_array(x):  # TODO
    return ivy.current_backend(x).is_native_sparse_array(x)
