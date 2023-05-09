# local
import ivy
from ivy.func_wrapper import inputs_to_native_arrays
from ivy.utils.exceptions import handle_exceptions


# helpers
def _verify_coo_components(indices=None, values=None, dense_shape=None):
    ivy.utils.assertions.check_all_or_any_fn(
        indices,
        values,
        dense_shape,
        fn=ivy.exists,
        type="all",
        message="indices, values and dense_shape must all be specified",
    )
    # coordinates style (COO), must be shaped (x, y)
    ivy.utils.assertions.check_equal(
        len(ivy.shape(indices)), 2, message="indices must be 2D"
    )
    ivy.utils.assertions.check_equal(
        len(ivy.shape(values)), 1, message="values must be 1D"
    )
    ivy.utils.assertions.check_equal(
        len(ivy.to_ivy_shape(dense_shape)),
        ivy.shape(indices)[0],
        message="shape and indices shape do not match",
    )
    # number of values must match number of coordinates
    ivy.utils.assertions.check_equal(
        ivy.shape(values)[0],
        ivy.shape(indices)[1],
        message="values and indices do not match",
    )
    for i in range(ivy.shape(indices)[0]):
        ivy.utils.assertions.check_less(
            indices[i],
            ivy.to_ivy_shape(dense_shape)[i],
            message="indices is larger than shape",
        )


def _verify_common_row_format_components(
    crow_indices=None, col_indices=None, values=None, dense_shape=None, format="csr"
):
    ivy.utils.assertions.check_all_or_any_fn(
        crow_indices,
        col_indices,
        values,
        dense_shape,
        fn=ivy.exists,
        type="all",
        message=(
            "crow_indices, col_indices, values and dense_shape must all be specified."
        ),
    )

    ivy.utils.assertions.check_equal(
        len(ivy.shape(crow_indices)), 1, message="crow_indices must be 1D."
    )
    ivy.utils.assertions.check_equal(
        len(ivy.shape(col_indices)), 1, message="col_indices must be 1D."
    )

    ivy.utils.assertions.check_equal(
        len(dense_shape),
        2,
        message=f"Only 2D arrays can be converted to {format.upper()} sparse arrays.",
    )

    ivy.utils.assertions.check_equal(
        ivy.shape(col_indices)[0],
        crow_indices[-1],
        message="size of col_indices does not match with last element of crow_indices",
    )

    # number of values must match number of coordinates
    ivy.utils.assertions.check_equal(
        ivy.shape(col_indices)[0],
        ivy.shape(values)[0],
        message="values and col_indices do not match",
    )

    # index in crow_indices must not exceed length of col_indices
    ivy.utils.assertions.check_less(
        crow_indices,
        ivy.shape(col_indices)[0],
        allow_equal=True,
        message="index in crow_indices does not match the number of col_indices",
    )


def _verify_csr_components(
    crow_indices=None, col_indices=None, values=None, dense_shape=None
):
    _verify_common_row_format_components(
        crow_indices=crow_indices,
        col_indices=col_indices,
        values=values,
        dense_shape=dense_shape,
        format="csr",
    )

    ivy.utils.assertions.check_equal(
        len(ivy.shape(values)), 1, message="values must be 1D."
    )
    # number of intervals must be equal to x in shape (x, y)
    ivy.utils.assertions.check_equal(ivy.shape(crow_indices)[0] - 1, dense_shape[0])

    ivy.utils.assertions.check_less(
        col_indices,
        dense_shape[1],
        message="index in col_indices does not match shape",
    )


def _verify_bsr_components(
    crow_indices=None, col_indices=None, values=None, dense_shape=None
):
    _verify_common_row_format_components(
        crow_indices=crow_indices,
        col_indices=col_indices,
        values=values,
        dense_shape=dense_shape,
        format="bsr",
    )
    ivy.utils.assertions.check_equal(
        len(ivy.shape(values)), 3, message="values must be 3D."
    )
    nrowblocks, ncolblocks = ivy.shape(values)[-2:]
    ivy.utils.assertions.check_equal(
        dense_shape[0] % nrowblocks,
        0,
        message="The number of rows of array must be divisible by that of block.",
    )
    ivy.utils.assertions.check_equal(
        dense_shape[1] % ncolblocks,
        0,
        message="The number of cols of array must be divisible by that of block.",
    )
    ivy.utils.assertions.check_equal(
        ivy.shape(crow_indices)[0] - 1, dense_shape[0] // nrowblocks
    )
    ivy.utils.assertions.check_less(
        col_indices,
        dense_shape[1] // ncolblocks,
        message="index in col_indices does not match shape",
    )


def _verify_common_column_format_components(
    ccol_indices=None, row_indices=None, values=None, dense_shape=None, format="csc"
):
    ivy.utils.assertions.check_all_or_any_fn(
        ccol_indices,
        row_indices,
        values,
        dense_shape,
        fn=ivy.exists,
        type="all",
        message=(
            "ccol_indices, row_indices, values and dense_shape must all be specified"
        ),
    )
    ivy.utils.assertions.check_equal(
        len(ivy.shape(ccol_indices)), 1, message="ccol_indices must be 1D"
    )
    ivy.utils.assertions.check_equal(
        len(ivy.shape(row_indices)), 1, message="row_indices must be 1D"
    )

    ivy.utils.assertions.check_equal(
        len(dense_shape),
        2,
        message=f"only 2D arrays can be converted to {format.upper()} sparse arrays",
    )
    # number of values must match number of coordinates
    ivy.utils.assertions.check_equal(
        ivy.shape(row_indices)[0],
        ivy.shape(values)[0],
        message="values and row_indices do not match",
    )
    # index in ccol_indices must not exceed length of row_indices
    ivy.utils.assertions.check_less(
        ccol_indices,
        ivy.shape(row_indices)[0],
        allow_equal=True,
        message="index in ccol_indices does not match the number of row_indices",
    )


def _verify_csc_components(
    ccol_indices=None, row_indices=None, values=None, dense_shape=None
):
    _verify_common_column_format_components(
        ccol_indices=ccol_indices,
        row_indices=row_indices,
        values=values,
        dense_shape=dense_shape,
        format="csc",
    )

    ivy.utils.assertions.check_equal(
        len(ivy.shape(values)), 1, message="values must be 1D"
    )
    # number of intervals must be equal to y in shape (x, y)
    ivy.utils.assertions.check_equal(ivy.shape(ccol_indices)[0] - 1, dense_shape[1])
    ivy.utils.assertions.check_less(
        row_indices,
        dense_shape[0],
        message="index in row_indices does not match shape",
    )


def _verify_bsc_components(
    ccol_indices=None, row_indices=None, values=None, dense_shape=None
):
    _verify_common_column_format_components(
        ccol_indices=ccol_indices,
        row_indices=row_indices,
        values=values,
        dense_shape=dense_shape,
        format="bsc",
    )
    ivy.utils.assertions.check_equal(
        len(ivy.shape(values)), 3, message="values must be 3D"
    )
    nrowblocks, ncolblocks = ivy.shape(values)[-2:]
    ivy.utils.assertions.check_equal(
        dense_shape[0] % nrowblocks,
        0,
        message="number of rows of array must be divisible by that of block.",
    )
    ivy.utils.assertions.check_equal(
        dense_shape[1] % ncolblocks,
        0,
        message="number of cols of array must be divisible by that of block.",
    )
    # number of intervals must be equal to y in shape (x, y)
    ivy.utils.assertions.check_equal(
        ivy.shape(ccol_indices)[0] - 1, dense_shape[1] // ncolblocks
    )
    ivy.utils.assertions.check_less(
        row_indices,
        dense_shape[0] // nrowblocks,
        message="index in row_indices does not match shape",
    )


def _is_data_not_indices_values_and_shape(
    data=None,
    coo_indices=None,
    crow_indices=None,
    col_indices=None,
    ccol_indices=None,
    row_indices=None,
    values=None,
    dense_shape=None,
    format=None,
):
    if data is not None:
        ivy.utils.assertions.check_all_or_any_fn(
            coo_indices,
            crow_indices,
            col_indices,
            ccol_indices,
            row_indices,
            values,
            dense_shape,
            format=format,
            fn=ivy.exists,
            type="any",
            limit=[0],
            message=(
                "Only specify data, coo_indices for COO format, crow_indices and"
                " col_indices for CSR and BSR, ccol_indices and row_indicesfor CSC and"
                " BSC."
            ),
        )
        return True
    return False


def _is_valid_format(
    coo_indices=None,
    crow_indices=None,
    col_indices=None,
    ccol_indices=None,
    row_indices=None,
    values=None,
    dense_shape=None,
    format="coo",
):
    valid_formats = ["coo", "csr", "csc", "csc", "bsc", "bsr"]

    if not isinstance(format, str) or not format.lower() in valid_formats:
        return False

    if format.endswith("o"):
        # format is coo
        return (
            ivy.exists(coo_indices)
            and ivy.exists(values)
            and ivy.exists(dense_shape)
            and crow_indices is None
            and col_indices is None
            and ccol_indices is None
            and row_indices is None
        )

    if format.endswith("r"):
        # format is either csr or bsr
        return (
            ivy.exists(crow_indices)
            and ivy.exists(col_indices)
            and ivy.exists(values)
            and ivy.exists(dense_shape)
            and coo_indices is None
            and ccol_indices is None
            and row_indices is None
        )
    # format is either csc or bsc
    return (
        ivy.exists(ccol_indices)
        and ivy.exists(row_indices)
        and ivy.exists(values)
        and ivy.exists(dense_shape)
        and coo_indices is None
        and crow_indices is None
        and col_indices is None
    )


class SparseArray:
    def __init__(
        self,
        data=None,
        *,
        coo_indices=None,
        crow_indices=None,
        col_indices=None,
        ccol_indices=None,
        row_indices=None,
        values=None,
        dense_shape=None,
        format=None,
    ):
        if _is_data_not_indices_values_and_shape(
            data,
            coo_indices,
            crow_indices,
            col_indices,
            ccol_indices,
            row_indices,
            values,
            dense_shape,
        ):
            self._init_data(data)
        elif _is_valid_format(
            coo_indices,
            crow_indices,
            col_indices,
            ccol_indices,
            row_indices,
            values,
            dense_shape,
            format=format,
        ):
            format = format.lower()

            if format == "coo":
                self._init_coo_components(coo_indices, values, dense_shape, format)
            elif format == "csr" or format == "bsr":
                self._init_compressed_row_components(
                    crow_indices, col_indices, values, dense_shape, format
                )
            else:
                print(format)
                self._init_compressed_column_components(
                    ccol_indices, row_indices, values, dense_shape, format
                )

        else:
            print(
                format,
                ccol_indices,
                row_indices,
                values,
                dense_shape,
                crow_indices,
                col_indices,
                values,
            )

            raise ivy.utils.exceptions.IvyException(
                "specify all coo components (coo_indices, values and "
                " dense_shape), all csr components (crow_indices, "
                "col_indices, values and dense_shape), all csc components "
                "(ccol_indices, row_indices, values and dense_shape). all "
                "bsc components (ccol_indices, row_indices, values and "
                "dense_shape), or all bsr components (crow_indices, "
                "col_indices, values and dense_shape)."
            )

    def _init_data(self, data):
        if ivy.is_ivy_sparse_array(data):
            self._data = data.data
            self._coo_indices = data.coo_indices
            self._crow_indices = data.crow_indices
            self._col_indices = data.col_indices
            self._ccol_indices = data.ccol_indices
            self._row_indices = data.row_indices
            self._values = data.values
            self._dense_shape = data.dense_shape
            self._format = data.format.lower()
        else:
            ivy.utils.assertions.check_true(
                ivy.is_native_sparse_array(data), message="not a native sparse array"
            )
            self._data = data
            self._native_sparse_array_to_indices_values_and_shape()

    def _native_sparse_array_to_indices_values_and_shape(self):
        indices, values, shape = ivy.native_sparse_array_to_indices_values_and_shape(
            self._data
        )

        if "coo_indices" in indices:
            self._coo_indices = ivy.array(indices["coo_indices"], dtype="int64")
            self._crow_indices = None
            self._col_indices = None
            self._ccol_indices = None
            self._row_indices = None

        elif "crow_indices" in indices and "col_indices" in indices:
            self._crow_indices = ivy.array(indices["crow_indices"], dtype="int64")
            self._col_indices = ivy.array(indices["col_indices"], dtype="int64")
            self._coo_indices = None
            self._ccol_indices = None
            self._row_indices = None

        else:
            self._ccol_indices = ivy.array(indices["ccol_indices"], dtype="int64")
            self._row_indices = ivy.array(indices["row_indices"], dtype="int64")
            self._coo_indices = None
            self._crow_indices = None
            self._col_indices = None

        self._values = ivy.array(values)
        self._dense_shape = ivy.Shape(shape)
        self._format = self._data.format.lower()

    def _init_coo_components(self, coo_indices, values, shape, format):
        coo_indices = ivy.array(coo_indices, dtype="int64")
        values = ivy.array(values)
        shape = ivy.Shape(shape)
        self._data = ivy.native_sparse_array(
            coo_indices=coo_indices, values=values, dense_shape=shape, format=format
        )
        self._coo_indices = coo_indices
        self._values = values
        self._dense_shape = shape
        self._format = format
        self._crow_indices = None
        self._col_indices = None
        self._ccol_indices = None
        self._row_indices = None

    def _init_compressed_row_components(
        self, crow_indices, col_indices, values, shape, format
    ):
        crow_indices = ivy.array(crow_indices, dtype="int64")
        col_indices = ivy.array(col_indices, dtype="int64")
        values = ivy.array(values)
        shape = ivy.Shape(shape)
        self._data = ivy.native_sparse_array(
            crow_indices=crow_indices,
            col_indices=col_indices,
            values=values,
            dense_shape=shape,
            format=format,
        )
        self._crow_indices = crow_indices
        self._col_indices = col_indices
        self._values = values
        self._dense_shape = shape
        self._format = format
        self._coo_indices = None
        self._ccol_indices = None
        self._row_indices = None

    def _init_compressed_column_components(
        self, ccol_indices, row_indices, values, shape, format
    ):
        ccol_indices = ivy.array(ccol_indices, dtype="int64")
        row_indices = ivy.array(row_indices, dtype="int64")
        values = ivy.array(values)
        shape = ivy.Shape(shape)
        self._data = ivy.native_sparse_array(
            ccol_indices=ccol_indices,
            row_indices=row_indices,
            values=values,
            dense_shape=shape,
            format=format,
        )
        self._ccol_indices = ccol_indices
        self._row_indices = row_indices
        self._values = values
        self._dense_shape = shape
        self._format = format
        self._coo_indices = None
        self._crow_indices = None
        self._col_indices = None

    # Properties #
    # -----------#

    @property
    def data(self):
        return self._data

    @property
    def coo_indices(self):
        return self._coo_indices

    @property
    def crow_indices(self):
        return self._crow_indices

    @property
    def col_indices(self):
        return self._col_indices

    @property
    def ccol_indices(self):
        return self._ccol_indices

    @property
    def row_indices(self):
        return self._row_indices

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

    @crow_indices.setter
    def crow_indices(self, indices):
        indices = ivy.array(indices, dtype="int64")
        if self._format == "csr":
            _verify_csr_components(
                crow_indices=indices,
                col_indices=self._col_indices,
                values=self._values,
                dense_shape=self._dense_shape,
            )
        else:
            _verify_bsr_components(
                crow_indices=indices,
                col_indices=self._col_indices,
                values=self._values,
                dense_shape=self._dense_shape,
            )
        self._crow_indices = indices

    @col_indices.setter
    def col_indices(self, indices):
        indices = ivy.array(indices, dtype="int64")
        if self._format == "csr":
            _verify_csr_components(
                crow_indices=indices,
                col_indices=self._col_indices,
                values=self._values,
                dense_shape=self._dense_shape,
            )
        else:
            _verify_bsr_components(
                crow_indices=indices,
                col_indices=self._col_indices,
                values=self._values,
                dense_shape=self._dense_shape,
            )
        self._col_indices = indices

    @ccol_indices.setter
    def ccol_indices(self, indices):
        indices = ivy.array(indices, dtype="int64")
        if self._format == "csc":
            _verify_csc_components(
                ccol_indices=indices,
                row_indices=self._row_indices,
                values=self._values,
                dense_shape=self._dense_shape,
            )
        else:
            _verify_bsc_components(
                ccol_indices=indices,
                row_indices=self._row_indices,
                values=self._values,
                dense_shape=self._dense_shape,
            )
        self._ccol_indices = indices

    @row_indices.setter
    def row_indices(self, indices):
        indices = ivy.array(indices, dtype="int64")
        if self._format == "csc":
            _verify_csc_components(
                ccol_indices=self._ccol_indices,
                row_indices=indices,
                values=self._values,
                dense_shape=self._dense_shape,
            )
        else:
            _verify_bsc_components(
                ccol_indices=self._ccol_indices,
                row_indices=indices,
                values=self._values,
                dense_shape=self._dense_shape,
            )
        self._row_indices = indices

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

    def _coo_to_dense_coordinates(self):
        all_coordinates = []
        for i in range(self._values.shape[0]):
            coordinate = ivy.gather(self._coo_indices, ivy.array([[i]]))
            coordinate = ivy.reshape(coordinate, (self._coo_indices.shape[0],))
            all_coordinates.append(coordinate.to_list())
        return all_coordinates

    def _csr_to_dense_coordinates(self):
        all_coordinates = []
        total_rows = self._dense_shape[0]
        all_rows = self._col_indices.to_list()
        all_cols = self._crow_indices.to_list()
        for row in range(total_rows):
            cols = all_rows[all_cols[row] : all_cols[row + 1]]
            for col in cols:
                all_coordinates.append([row, col])
        return all_coordinates

    def _csc_to_dense_coordinates(self):
        # CSC sparse array
        all_coordinates = []
        total_rows = self._dense_shape[1]
        all_cols = self._row_indices.to_list()
        all_rows = self._ccol_indices.to_list()
        for col in range(total_rows):
            rows = all_cols[all_rows[col] : all_rows[col + 1]]
            for row in rows:
                all_coordinates.append([row, col])
        return all_coordinates

    def _bsr_to_dense_coordinates(self):
        all_coordinates = []
        total_rows = self._dense_shape[0]
        all_rows = self._crow_indices.to_list()
        all_cols = self._col_indices.to_list()

        nblockrows, nblockcols = self._values.shape[-2:]

        for row in range(total_rows // nblockrows):
            cols = all_cols[all_rows[row] : all_rows[row + 1]]
            for col in cols:
                for col_index in range(nblockcols):
                    for row_index in range(nblockrows):
                        all_coordinates.append(
                            [
                                nblockrows * row + row_index,
                                nblockcols * col + col_index,
                            ]
                        )
        return all_coordinates

    def _bsc_to_dense_coordinates(self):
        all_coordinates = []
        total_cols = self._dense_shape[1]
        all_rows = self._row_indices.to_list()
        all_cols = self._ccol_indices.to_list()

        nblockrows, nblockcols = self._values.shape[-2:]

        for col in range(total_cols // nblockcols):
            rows = all_rows[all_cols[col] : all_cols[col + 1]]
            for row in rows:
                for col_index in range(nblockcols):
                    for row_index in range(nblockrows):
                        all_coordinates.append(
                            [
                                nblockrows * row + row_index,
                                nblockcols * col + col_index,
                            ]
                        )
        return all_coordinates

    def to_dense_array(self, *, native=False):
        if self._format == "coo":
            all_coordinates = self._coo_to_dense_coordinates()
        elif self._format == "csr":
            all_coordinates = self._csr_to_dense_coordinates()
        elif self._format == "csc":
            all_coordinates = self._csc_to_dense_coordinates()
        elif self._format == "bsc":
            all_coordinates = self._bsc_to_dense_coordinates()
        else:
            all_coordinates = self._bsr_to_dense_coordinates()

        # make dense array
        ret = ivy.scatter_nd(
            ivy.array(all_coordinates),
            ivy.flatten(self._values),
            ivy.array(self._dense_shape),
        )
        return ret.to_native() if native else ret


class NativeSparseArray:
    pass


def is_ivy_sparse_array(x):
    return isinstance(x, ivy.SparseArray)


@handle_exceptions
@inputs_to_native_arrays
def is_native_sparse_array(x):
    return ivy.current_backend().is_native_sparse_array(x)


@handle_exceptions
@inputs_to_native_arrays
def native_sparse_array(
    data=None,
    *,
    coo_indices=None,
    crow_indices=None,
    col_indices=None,
    ccol_indices=None,
    row_indices=None,
    values=None,
    dense_shape=None,
    format=None,
):
    return ivy.current_backend().native_sparse_array(
        data,
        coo_indices=coo_indices,
        crow_indices=crow_indices,
        col_indices=col_indices,
        ccol_indices=ccol_indices,
        row_indices=row_indices,
        values=values,
        dense_shape=dense_shape,
        format=format,
    )


@handle_exceptions
def native_sparse_array_to_indices_values_and_shape(x):
    return ivy.current_backend().native_sparse_array_to_indices_values_and_shape(x)
