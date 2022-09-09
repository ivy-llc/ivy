import ivy
from ivy.func_wrapper import inputs_to_native_arrays


# helpers
def _verify_coo_components(*, indices=None, values=None, dense_shape=None):
    assert (
        ivy.exists(indices) and ivy.exists(values) and ivy.exists(dense_shape)
    ), "indices, values and dense_shape must all be specified"
    # coordinates style (COO), must be shaped (x, y)
    assert len(ivy.shape(indices)) == 2, "indices must be 2D"
    assert (
        len(ivy.to_ivy_shape(dense_shape)) == ivy.shape(indices)[0]
    ), "shape and indices shape do not match"
    # number of values must match number of coordinates
    assert (
        ivy.shape(values)[0] == ivy.shape(indices)[1]
    ), "values and indices do not match"
    for i in range(ivy.shape(indices)[0]):
        assert ivy.all(
            ivy.less(indices[i], ivy.to_ivy_shape(dense_shape)[i])
        ), "indices is larger than shape"


def _is_data_not_indices_values_and_shape(
    data=None, indices=None, values=None, dense_shape=None
):
    if data is not None:
        if ivy.exists(indices) or ivy.exists(values) or ivy.exists(dense_shape):
            raise Exception(
                "only specify either data or or all three components: \
                indices, values and dense_shape"
            )
        return True
    return False


class SparseArray:
    def __init__(self, data=None, *, indices=None, values=None, dense_shape=None):
        if _is_data_not_indices_values_and_shape(data, indices, values, dense_shape):
            self._init_data(data)
        elif ivy.exists(indices) and ivy.exists(values) and ivy.exists(dense_shape):
            self._init_coo_components(indices, values, dense_shape)
        # TODO: to add csr
        else:
            raise Exception("indices, values and dense_shape must all be specified")

    def _init_data(self, data):
        if ivy.is_ivy_sparse_array(data):
            self._data = data.data
            self._indices = data.indices
            self._values = data.values
            self._dense_shape = data.dense_shape
        else:
            assert ivy.is_native_sparse_array(data), "not a native sparse array"
            self._data = data
            self._native_sparse_array_to_indices_values_and_shape()

    def _native_sparse_array_to_indices_values_and_shape(self):
        indices, values, shape = ivy.native_sparse_array_to_indices_values_and_shape(
            self._data
        )
        self._indices = ivy.array(indices, dtype="int64")
        self._values = ivy.array(values)
        self._dense_shape = ivy.Shape(shape)

    def _init_coo_components(self, indices, values, shape):
        self._data = ivy.native_sparse_array(
            indices=indices, values=values, dense_shape=shape
        )
        self._indices = ivy.array(indices, dtype="int64")
        self._values = ivy.array(values)
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
    def data(self, data):
        self._init_data(data)

    @indices.setter
    def indices(self, indices):
        indices = ivy.array(indices, dtype="int64")
        _verify_coo_components(
            indices=indices, values=self._values, dense_shape=self._dense_shape
        )
        self._indices = indices

    @values.setter
    def values(self, values):
        values = ivy.array(values)
        _verify_coo_components(
            indices=self._indices, values=values, dense_shape=self._dense_shape
        )
        self._values = values

    @dense_shape.setter
    def dense_shape(self, dense_shape):
        dense_shape = ivy.Shape(dense_shape)
        _verify_coo_components(
            indices=self._indices, values=self._values, dense_shape=dense_shape
        )
        self._dense_shape = dense_shape

    # Instance Methods #
    # ---------------- #

    def to_dense_array(self, *, native=False):
        all_coordinates = []
        for i in range(self._values.shape[0]):
            coordinate = ivy.gather(self._indices, ivy.array([[i]]))
            coordinate = ivy.reshape(coordinate, (self._indices.shape[0],))
            all_coordinates.append(coordinate.to_list())
        ret = ivy.scatter_nd(
            ivy.array(all_coordinates), self._values, ivy.array(self._dense_shape)
        )
        return ret.to_native() if native else ret


class NativeSparseArray:
    pass


def is_ivy_sparse_array(x):
    return isinstance(x, ivy.SparseArray)


@inputs_to_native_arrays
def is_native_sparse_array(x):
    return ivy.current_backend().is_native_sparse_array(x)


@inputs_to_native_arrays
def native_sparse_array(data=None, *, indices=None, values=None, dense_shape=None):
    return ivy.current_backend().native_sparse_array(
        data, indices=indices, values=values, dense_shape=dense_shape
    )


def native_sparse_array_to_indices_values_and_shape(x):
    return ivy.current_backend().native_sparse_array_to_indices_values_and_shape(x)
