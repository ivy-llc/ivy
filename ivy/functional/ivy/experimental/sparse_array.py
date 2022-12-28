# local
import ivy
from ivy.func_wrapper import inputs_to_native_arrays
from ivy.exceptions import handle_exceptions


# helpers
def _verify_coo_components(*, indices=None, values=None, dense_shape=None):
    ivy.assertions.check_all_or_any_fn(
        indices,
        values,
        dense_shape,
        fn=ivy.exists,
        type="all",
        message="indices, values and dense_shape must all be specified",
    )
    # coordinates style (COO), must be shaped (x, y)
    ivy.assertions.check_equal(len(ivy.shape(indices)), 2, message="indices must be 2D")
    ivy.assertions.check_equal(len(ivy.shape(values)), 1, message="values must be 1D")
    ivy.assertions.check_equal(
        len(ivy.to_ivy_shape(dense_shape)),
        ivy.shape(indices)[0],
        message="shape and indices shape do not match",
    )
    # number of values must match number of coordinates
    ivy.assertions.check_equal(
        ivy.shape(values)[0],
        ivy.shape(indices)[1],
        message="values and indices do not match",
    )
    for i in range(ivy.shape(indices)[0]):
        ivy.assertions.check_less(
            indices[i],
            ivy.to_ivy_shape(dense_shape)[i],
            message="indices is larger than shape",
        )


def _verify_csr_components(
    *, crow_indices=None, col_indices=None, values=None, dense_shape=None
):
    ivy.assertions.check_all_or_any_fn(
        crow_indices,
        col_indices,
        values,
        dense_shape,
        fn=ivy.exists,
        type="all",
        message="crow_indices, col_indices, values and dense_shape must all \
        be specified",
    )
    ivy.assertions.check_equal(
        len(ivy.shape(crow_indices)), 1, message="crow_indices must be 1D"
    )
    ivy.assertions.check_equal(
        len(ivy.shape(col_indices)), 1, message="col_indices must be 1D"
    )
    ivy.assertions.check_equal(len(ivy.shape(values)), 1, message="values must be 1D")
    ivy.assertions.check_equal(
        len(dense_shape),
        2,
        message="only 2D arrays can be converted to CSR sparse arrays",
    )
    # number of intervals must be equal to x in shape (x, y)
    ivy.assertions.check_equal(ivy.shape(crow_indices)[0] - 1, dense_shape[0])
    # index in col_indices must not exceed y in shape (x, y)
    ivy.assertions.check_less(
        col_indices, dense_shape[1], message="index in col_indices does not match shape"
    )
    # number of values must match number of coordinates
    ivy.assertions.check_equal(
        ivy.shape(col_indices)[0],
        ivy.shape(values)[0],
        message="values and col_indices do not match",
    )
    # index in crow_indices must not exceed length of col_indices
    ivy.assertions.check_less(
        crow_indices,
        ivy.shape(col_indices)[0],
        allow_equal=True,
        message="index in crow_indices does not match the number of col_indices",
    )


def _verify_csc_components(
    *, ccol_indices=None, row_indices=None, values=None, dense_shape=None
):
    ivy.assertions.check_all_or_any_fn(
        ccol_indices,
        row_indices,
        values,
        dense_shape,
        fn=ivy.exists,
        type="all",
        message="ccol_indices, row_indices, values and dense_shape must all \
        be specified",
    )
    ivy.assertions.check_equal(
        len(ivy.shape(ccol_indices)), 1, message="ccol_indices must be 1D"
    )
    ivy.assertions.check_equal(
        len(ivy.shape(row_indices)), 1, message="row_indices must be 1D"
    )
    ivy.assertions.check_equal(len(ivy.shape(values)), 1, message="values must be 1D")
    ivy.assertions.check_equal(
        len(dense_shape),
        2,
        message="only 2D arrays can be converted to CSC sparse arrays",
    )
    # number of intervals must be equal to y in shape (x, y)
    ivy.assertions.check_equal(ivy.shape(ccol_indices)[0] - 1, dense_shape[1])
    # index in row_indices must not exceed x in shape (x, y)
    ivy.assertions.check_less(
        row_indices, dense_shape[0], message="index in row_indices does not match shape"
    )
    # number of values must match number of coordinates
    ivy.assertions.check_equal(
        ivy.shape(row_indices)[0],
        ivy.shape(values)[0],
        message="values and row_indices do not match",
    )
    # index in ccol_indices must not exceed length of row_indices
    ivy.assertions.check_less(
        ccol_indices,
        ivy.shape(row_indices)[0],
        allow_equal=True,
        message="index in ccol_indices does not match the number of row_indices",
    )


def _is_data_not_indices_values_and_shape(
    data=None,
    coo_indices=None,
    csr_crow_indices=None,
    csr_col_indices=None,
    csc_ccol_indices=None,
    csc_row_indices=None,
    values=None,
    dense_shape=None,
):
    if data is not None:
        ivy.assertions.check_all_or_any_fn(
            coo_indices,
            csr_crow_indices,
            csr_col_indices,
            csc_ccol_indices,
            csc_row_indices,
            values,
            dense_shape,
            fn=ivy.exists,
            type="any",
            limit=[0],
            message="only specify data, all coo components (coo_indices, values \
            and dense_shape), or all csr components (csr_crow_indices, \
            csr_col_indices, values and dense_shape), or all csc components \
                (csc_ccol_indices, csc_row_indices, values and dense_shape).",
        )
        return True
    return False


def _is_coo(
    coo_indices=None,
    csr_crow_indices=None,
    csr_col_indices=None,
    csc_ccol_indices=None,
    csc_row_indices=None,
    values=None,
    dense_shape=None,
):
    if (
        ivy.exists(coo_indices)
        and ivy.exists(values)
        and ivy.exists(dense_shape)
        and csr_crow_indices is None
        and csr_col_indices is None
        and csc_ccol_indices is None
        and csc_row_indices is None
    ):
        return True
    return False


def _is_csr(
    coo_indices=None,
    csr_crow_indices=None,
    csr_col_indices=None,
    csc_ccol_indices=None,
    csc_row_indices=None,
    values=None,
    dense_shape=None,
):
    if (
        ivy.exists(csr_crow_indices)
        and ivy.exists(csr_col_indices)
        and ivy.exists(values)
        and ivy.exists(dense_shape)
        and coo_indices is None
        and csc_ccol_indices is None
        and csc_row_indices is None
    ):
        return True

    return False


def _is_csc(
    coo_indices=None,
    csr_crow_indices=None,
    csr_col_indices=None,
    csc_ccol_indices=None,
    csc_row_indices=None,
    values=None,
    dense_shape=None,
):
    if (
        ivy.exists(csc_ccol_indices)
        and ivy.exists(csc_row_indices)
        and ivy.exists(values)
        and ivy.exists(dense_shape)
        and coo_indices is None
        and csr_crow_indices is None
        and csr_col_indices is None
    ):
        return True

    return False


class SparseArray:
    def __init__(
        self,
        data=None,
        *,
        coo_indices=None,
        csr_crow_indices=None,
        csr_col_indices=None,
        csc_ccol_indices=None,
        csc_row_indices=None,
        values=None,
        dense_shape=None,
    ):
        if _is_data_not_indices_values_and_shape(
            data,
            coo_indices,
            csr_crow_indices,
            csr_col_indices,
            csc_ccol_indices,
            csc_row_indices,
            values,
            dense_shape,
        ):
            self._init_data(data)
        elif _is_coo(
            coo_indices,
            csr_crow_indices,
            csr_col_indices,
            csc_ccol_indices,
            csc_row_indices,
            values,
            dense_shape,
        ):
            self._init_coo_components(coo_indices, values, dense_shape)
        elif _is_csr(
            coo_indices,
            csr_crow_indices,
            csr_col_indices,
            csc_ccol_indices,
            csc_row_indices,
            values,
            dense_shape,
        ):
            self._init_csr_components(
                csr_crow_indices, csr_col_indices, values, dense_shape
            )
        elif _is_csc(
            coo_indices,
            csr_crow_indices,
            csr_col_indices,
            csc_ccol_indices,
            csc_row_indices,
            values,
            dense_shape,
        ):
            self._init_csc_components(
                csc_ccol_indices, csc_row_indices, values, dense_shape
            )
        else:
            raise ivy.exceptions.IvyException(
                "specify all coo components (coo_indices, values \
            and dense_shape), or all csr components (csr_crow_indices, \
            csr_col_indices, values and dense_shape), or all csc components \
                (csc_ccol_indices, csc_row_indices, values and dense_shape)."
            )

    def _init_data(self, data):
        if ivy.is_ivy_sparse_array(data):
            self._data = data.data
            self._coo_indices = data.coo_indices
            self._csr_crow_indices = data.csr_crow_indices
            self._csr_col_indices = data.csr_col_indices
            self._csc_ccol_indices = data.csc_ccol_indices
            self._csc_row_indices = data.csc_row_indices
            self._values = data.values
            self._dense_shape = data.dense_shape
        else:
            ivy.assertions.check_true(
                ivy.is_native_sparse_array(data), message="not a native sparse array"
            )
            self._data = data
            self._native_sparse_array_to_indices_values_and_shape()

    def _native_sparse_array_to_indices_values_and_shape(self):
        indices, values, shape = ivy.native_sparse_array_to_indices_values_and_shape(
            self._data
        )

        if "indices" in indices:
            self._coo_indices = ivy.array(indices["indices"], dtype="int64")
            self._csr_crow_indices = None
            self._csr_col_indices = None
            self._csc_ccol_indices = None
            self._csc_row_indices = None
        elif "crow_indices" in indices and "col_indices" in indices:
            self._csr_crow_indices = ivy.array(indices["crow_indices"], dtype="int64")
            self._csr_col_indices = ivy.array(indices["col_indices"], dtype="int64")
            self._coo_indices = None
            self._csc_ccol_indices = None
            self._csc_row_indices = None
        else:
            self._csc_ccol_indices = ivy.array(indices["ccol_indices"], dtype="int64")
            self._csc_row_indices = ivy.array(indices["row_indices"], dtype="int64")
            self._coo_indices = None
            self._csr_crow_indices = None
            self._csr_col_indices = None

        self._values = ivy.array(values)
        self._dense_shape = ivy.Shape(shape)

    def _init_coo_components(self, coo_indices, values, shape):
        coo_indices = ivy.array(coo_indices, dtype="int64")
        values = ivy.array(values)
        shape = ivy.Shape(shape)
        self._data = ivy.native_sparse_array(
            coo_indices=coo_indices, values=values, dense_shape=shape
        )
        self._coo_indices = coo_indices
        self._values = values
        self._dense_shape = shape
        self._csr_crow_indices = None
        self._csr_col_indices = None
        self._csc_ccol_indices = None
        self._csc_row_indices = None

    def _init_csr_components(self, csr_crow_indices, csr_col_indices, values, shape):
        csr_crow_indices = ivy.array(csr_crow_indices, dtype="int64")
        csr_col_indices = ivy.array(csr_col_indices, dtype="int64")
        values = ivy.array(values)
        shape = ivy.Shape(shape)
        self._data = ivy.native_sparse_array(
            csr_crow_indices=csr_crow_indices,
            csr_col_indices=csr_col_indices,
            values=values,
            dense_shape=shape,
        )
        self._csr_crow_indices = csr_crow_indices
        self._csr_col_indices = csr_col_indices
        self._values = values
        self._dense_shape = shape
        self._coo_indices = None
        self._csc_ccol_indices = None
        self._csc_row_indices = None

    def _init_csc_components(self, csc_ccol_indices, csc_row_indices, values, shape):
        csc_ccol_indices = ivy.array(csc_ccol_indices, dtype="int64")
        csc_row_indices = ivy.array(csc_row_indices, dtype="int64")
        values = ivy.array(values)
        shape = ivy.Shape(shape)
        self._data = ivy.native_sparse_array(
            csc_ccol_indices=csc_ccol_indices,
            csc_row_indices=csc_row_indices,
            values=values,
            dense_shape=shape,
        )
        self._csc_ccol_indices = csc_ccol_indices
        self._csc_row_indices = csc_row_indices
        self._values = values
        self._dense_shape = shape
        self._coo_indices = None
        self._csr_crow_indices = None
        self._csr_col_indices = None

    # Properties #
    # -----------#

    @property
    def data(self):
        return self._data

    @property
    def coo_indices(self):
        return self._coo_indices

    @property
    def csr_crow_indices(self):
        return self._csr_crow_indices

    @property
    def csr_col_indices(self):
        return self._csr_col_indices

    @property
    def csc_ccol_indices(self):
        return self._csc_ccol_indices

    @property
    def csc_row_indices(self):
        return self._csc_row_indices

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

    @coo_indices.setter
    def coo_indices(self, indices):
        indices = ivy.array(indices, dtype="int64")
        _verify_coo_components(
            indices=indices, values=self._values, dense_shape=self._dense_shape
        )
        self._coo_indices = indices

    @csr_crow_indices.setter
    def csr_crow_indices(self, indices):
        indices = ivy.array(indices, dtype="int64")
        _verify_csr_components(
            crow_indices=indices,
            col_indices=self._csr_col_indices,
            values=self._values,
            dense_shape=self._dense_shape,
        )
        self._csr_crow_indices = indices

    @csr_col_indices.setter
    def csr_col_indices(self, indices):
        indices = ivy.array(indices, dtype="int64")
        _verify_csr_components(
            crow_indices=self._csr_crow_indices,
            col_indices=indices,
            values=self._values,
            dense_shape=self._dense_shape,
        )
        self._csr_col_indices = indices

    @csc_ccol_indices.setter
    def csc_ccol_indices(self, indices):
        indices = ivy.array(indices, dtype="int64")
        _verify_csc_components(
            ccol_indices=indices,
            row_indices=self._csc_row_indices,
            values=self._values,
            dense_shape=self._dense_shape,
        )
        self._csc_ccol_indices = indices

    @csc_row_indices.setter
    def csc_row_indices(self, indices):
        indices = ivy.array(indices, dtype="int64")
        _verify_csc_components(
            ccol_indices=self._csc_ccol_indices,
            row_indices=indices,
            values=self._values,
            dense_shape=self._dense_shape,
        )
        self._csc_row_indices = indices

    @values.setter
    def values(self, values):
        values = ivy.array(values)
        _verify_coo_components(
            indices=self._coo_indices, values=values, dense_shape=self._dense_shape
        )
        self._values = values

    @dense_shape.setter
    def dense_shape(self, dense_shape):
        dense_shape = ivy.Shape(dense_shape)
        _verify_coo_components(
            indices=self._coo_indices, values=self._values, dense_shape=dense_shape
        )
        self._dense_shape = dense_shape

    # Instance Methods #
    # ---------------- #

    def to_dense_array(self, *, native=False):
        all_coordinates = []

        if self._coo_indices is not None:
            # COO sparse array
            for i in range(self._values.shape[0]):
                coordinate = ivy.gather(self._coo_indices, ivy.array([[i]]))
                coordinate = ivy.reshape(coordinate, (self._coo_indices.shape[0],))
                all_coordinates.append(coordinate.to_list())
        elif self._csr_crow_indices is not None and self._csr_col_indices is not None:
            # CSR sparse array
            total_rows = self._dense_shape[0]
            all_cols = self._csr_col_indices.to_list()
            all_rows = self._csr_crow_indices.to_list()
            for row in range(total_rows):
                cols = all_cols[all_rows[row] : all_rows[row + 1]]
                for col in cols:
                    all_coordinates.append([row, col])
        else:
            # CSC sparse array
            total_cols = self._dense_shape[1]
            all_rows = self._csc_row_indices.to_list()
            all_cols = self._csc_ccol_indices.to_list()
            for col in range(total_cols):
                rows = all_rows[all_cols[col] : all_cols[col + 1]]
                for row in rows:
                    all_coordinates.append([row, col])

        # make dense array
        ret = ivy.scatter_nd(
            ivy.array(all_coordinates), self._values, ivy.array(self._dense_shape)
        )
        return ret.to_native() if native else ret


class NativeSparseArray:
    pass


def is_ivy_sparse_array(x):
    return isinstance(x, ivy.SparseArray)


@inputs_to_native_arrays
@handle_exceptions
def is_native_sparse_array(x):
    return ivy.current_backend().is_native_sparse_array(x)


@inputs_to_native_arrays
@handle_exceptions
def native_sparse_array(
    data=None,
    *,
    coo_indices=None,
    csr_crow_indices=None,
    csr_col_indices=None,
    csc_ccol_indices=None,
    csc_row_indices=None,
    values=None,
    dense_shape=None,
):
    return ivy.current_backend().native_sparse_array(
        data,
        coo_indices=coo_indices,
        csr_crow_indices=csr_crow_indices,
        csr_col_indices=csr_col_indices,
        csc_ccol_indices=csc_ccol_indices,
        csc_row_indices=csc_row_indices,
        values=values,
        dense_shape=dense_shape,
    )


@handle_exceptions
def native_sparse_array_to_indices_values_and_shape(x):
    return ivy.current_backend().native_sparse_array_to_indices_values_and_shape(x)
