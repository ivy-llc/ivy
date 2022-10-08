from typing import Optional, Union
import ivy
from ivy.func_wrapper import (
    handle_out_argument,
    to_native_arrays_and_back,
    handle_nestable,
    integer_arrays_to_float,
    inputs_to_native_arrays,
)
from ivy.exceptions import handle_exceptions
from numpy import prod


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


def _is_data_not_indices_values_and_shape(
    data=None,
    coo_indices=None,
    csr_crow_indices=None,
    csr_col_indices=None,
    values=None,
    dense_shape=None,
):
    if data is not None:
        ivy.assertions.check_all_or_any_fn(
            coo_indices,
            csr_crow_indices,
            csr_col_indices,
            values,
            dense_shape,
            fn=ivy.exists,
            type="any",
            limit=[0],
            message="only specify either data, all coo components (coo_indices, values \
            and dense_shape), or all csr components (csr_crow_indices, \
            csr_col_indices, values and dense_shape)",
        )
        return True
    return False


def _is_coo_not_csr(
    coo_indices=None,
    csr_crow_indices=None,
    csr_col_indices=None,
    values=None,
    dense_shape=None,
):
    if (
        ivy.exists(coo_indices)
        and ivy.exists(values)
        and ivy.exists(dense_shape)
        and csr_crow_indices is None
        and csr_col_indices is None
    ):
        return True
    elif (
        ivy.exists(csr_crow_indices)
        and ivy.exists(csr_col_indices)
        and ivy.exists(values)
        and ivy.exists(dense_shape)
        and coo_indices is None
    ):
        return False
    else:
        raise ivy.exceptions.IvyException(
            "specify either all coo components (coo_indices, values \
            and dense_shape), or all csr components (csr_crow_indices, \
            csr_col_indices, values and dense_shape)"
        )


class SparseArray:
    def __init__(
        self,
        data=None,
        *,
        coo_indices=None,
        csr_crow_indices=None,
        csr_col_indices=None,
        values=None,
        dense_shape=None,
    ):
        if _is_data_not_indices_values_and_shape(
            data, coo_indices, csr_crow_indices, csr_col_indices, values, dense_shape
        ):
            self._init_data(data)
        elif _is_coo_not_csr(
            coo_indices, csr_crow_indices, csr_col_indices, values, dense_shape
        ):
            self._init_coo_components(coo_indices, values, dense_shape)
        else:
            self._init_csr_components(
                csr_crow_indices, csr_col_indices, values, dense_shape
            )

    def _init_data(self, data):
        if ivy.is_ivy_sparse_array(data):
            self._data = data.data
            self._coo_indices = data.coo_indices
            self._csr_crow_indices = data.csr_crow_indices
            self._csr_col_indices = data.csr_col_indices
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
        if isinstance(indices, list):
            self._csr_crow_indices = ivy.array(indices[0], dtype="int64")
            self._csr_col_indices = ivy.array(indices[1], dtype="int64")
            self._coo_indices = None
        else:
            self._coo_indices = ivy.array(indices, dtype="int64")
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
        else:
            # CSR sparse array
            row = 0
            total_rows = self._dense_shape[0]
            all_cols = self._csr_col_indices.to_list()
            all_rows = self._csr_crow_indices.to_list()
            while row < total_rows:
                cols = all_cols[all_rows[row] : all_rows[row + 1]]
                for col in cols:
                    all_coordinates.append([row, col])
                row += 1
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
    values=None,
    dense_shape=None,
):
    return ivy.current_backend().native_sparse_array(
        data,
        coo_indices=coo_indices,
        csr_crow_indices=csr_crow_indices,
        csr_col_indices=csr_col_indices,
        values=values,
        dense_shape=dense_shape,
    )


@handle_exceptions
def native_sparse_array_to_indices_values_and_shape(x):
    return ivy.current_backend().native_sparse_array_to_indices_values_and_shape(x)


@integer_arrays_to_float
@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
def sinc(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Calculates an implementation-dependent approximation of the principal value of
    the normalized sinc function, having domain ``(-infinity, +infinity)`` and
    codomain ``[-0.217234, 1]``, for each element ``x_i`` of the input array ``x``.
    Each element ``x_i`` is assumed to be expressed in radians.

    **Special cases**

    For floating-point operands,

    - If x_i is NaN, the result is NaN.
    - If ``x_i`` is ``0``, the result is ``1``.
    - If ``x_i`` is either ``+infinity`` or ``-infinity``, the result is ``NaN``.

    Parameters
    ----------
    x
        input array. Should have a floating-point data type.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the normalized sinc function of each element in x.
        The returned array must have a floating-point data type determined
        by :ref:`type-promotion`.

    Examples
    --------
    With :code:`ivy.Array` input:

    >>> x = ivy.array([0.5, 1.5, 2.5, 3.5])
    >>> y = x.sinc()
    >>> print(y)
    ivy.array([0.637,-0.212,0.127,-0.0909])

    >>> x = ivy.array([1.5, 0.5, -1.5])
    >>> y = ivy.zeros(3)
    >>> ivy.sinc(x, out=y)
    >>> print(y)
    ivy.array(([-0.212,0.637,-0.212])


    With :code:`ivy.NativeArray` input:

    >>> x = ivy.array([0.5, 1.5, 2.5, 3.5])
    >>> y = ivy.sinc(x)
    >>> print(y)
    ivy.array([0.637,-0.212,0.127,-0.0909])

    With :code:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0.5, 1.5, 2.5]),\
                          b=ivy.array([3.5, 4.5, 5.5]))
    >>> y = x.sinc()
    >>> print(y)
    {
        a: ivy.array([0.637,-0.212,0.127]),
        b: ivy.array([-0.0909,0.0707,-0.0579])
    }
    """
    return ivy.current_backend(x).sinc(x, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
def flatten(
        x: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        start_dim: int = None,
        end_dim: int = None,
        out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Flattens input by reshaping it into a one-dimensional tensor.
        If start_dim or end_dim are passed, only dimensions starting
        with start_dim and ending with end_dim are flattened.
        The order of elements in input is unchanged.

    Parameters
    ----------
    x
        input array to flatten.
    start_dim
        first dim to flatten. If not set, defaults to 0.
    end_dim
        last dim to flatten. If not set, defaults to -1.

    Returns
    -------
    ret
        the flattened array over the specified dimensions.

    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.manipulation_functions.concat.html>`_ # noqa
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = np.array([1,2], [3,4])
    >>> ivy.flatten(x)
    ivy.array([1, 2, 3, 4])

    >>> x = np.array(
        [[[[ 5,  5,  0,  6],
         [17, 15, 11, 16],
         [ 6,  3, 13, 12]],

        [[ 6, 18, 10,  4],
         [ 5,  1, 17,  3],
         [14, 14, 18,  6]]],


       [[[12,  0,  1, 13],
         [ 8,  7,  0,  3],
         [19, 12,  6, 17]],

        [[ 4, 15,  6, 15],
         [ 0,  5, 17,  9],
         [ 9,  3,  6, 19]]],


       [[[17, 13, 11, 16],
         [ 4, 18, 17,  4],
         [10, 10,  9,  1]],

        [[19, 17, 13, 10],
         [ 4, 19, 16, 17],
         [ 2, 12,  8, 14]]]]
         )
    >>> ivy.flatten(x, start_dim = 1, end_dim = 2)
    ivy.array(
        [[[ 5,  5,  0,  6],
          [17, 15, 11, 16],
          [ 6,  3, 13, 12],
          [ 6, 18, 10,  4],
          [ 5,  1, 17,  3],
          [14, 14, 18,  6]],

         [[12,  0,  1, 13],
          [ 8,  7,  0,  3],
          [19, 12,  6, 17],
          [ 4, 15,  6, 15],
          [ 0,  5, 17,  9],
          [ 9,  3,  6, 19]],

         [[17, 13, 11, 16],
          [ 4, 18, 17,  4],
          [10, 10,  9,  1],
          [19, 17, 13, 10],
          [ 4, 19, 16, 17],
          [ 2, 12,  8, 14]]]))

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
    ...                   b=ivy.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]]))
    >>> ivy.flatten(x)
    [{
        a: ivy.array([1, 2, 3, 4, 5, 6, 7, 8])
        b: ivy.array([9, 10, 11, 12, 13, 14, 15, 16])
    }]
    """
    if start_dim == end_dim and len(x.shape) != 0:
        return x
    if start_dim not in range(- len(x.shape), len(x.shape)):
        raise IndexError(
            f"Dimension out of range (expected to be in range of\
            {[-len(x.shape), len(x.shape) - 1]}, but got {start_dim}"
        )
    if end_dim not in range(- len(x.shape), len(x.shape)):
        raise IndexError(
            f"Dimension out of range (expected to be in range of\
            {[-len(x.shape), len(x.shape) - 1]}, but got {end_dim}"
        )
    if start_dim is None:
        start_dim = 0
    if end_dim is None:
        end_dim = x.shape[-1]
    if start_dim < 0:
        start_dim = len(x.shape) + start_dim
    if end_dim < 0:
        end_dim = len(x.shape) + end_dim

    x_shape = x.shape
    new_shape = tuple(x_shape[:start_dim]) + (
        int(prod(x_shape[start_dim:end_dim + 1])),) + tuple(x_shape[end_dim + 1:])
    return ivy.reshape(x, new_shape)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
def vorbis_window(
    window_length: Union[ivy.Array, ivy.NativeArray],
    *,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Returns an array that contains a vorbis power complementary window
    of size window_length.
    Parameters
    ----------
    window_length
        the length of the vorbis window.
    dtype
        data type of the returned array. By default float32.
    out
        optional output array, for writing the result to.
    Returns
    -------
    ret
        Input array with the vorbis window.
    Examples
    --------
    >>> ivy.vorbis_window(3)
    ivy.array([0.38268346, 1. , 0.38268352])
    >>> ivy.vorbis_window(5)
    ivy.array(array([0.14943586, 0.8563191 , 1. , 0.8563191, 0.14943568])
    """
    return ivy.current_backend().vorbis_window(window_length, dtype=dtype, out=out)
