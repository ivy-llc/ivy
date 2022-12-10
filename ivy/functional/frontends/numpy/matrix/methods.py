# global
import ivy
import numpy as np

# local
from ivy.func_wrapper import from_zero_dim_arrays_to_float


class matrix:
    def __init__(self, data, dtype=None, copy=True):
        self._init_data(data, dtype)

    def _init_data(self, data, dtype):
        if isinstance(data, str):
            self._process_str_data(data, dtype)
        elif isinstance(data, (list, np.ndarray)) or ivy.is_array(data):
            if ivy.is_array(data) and dtype is None:
                dtype = data.dtype
            data = ivy.array(data, dtype=dtype)
            self._data = data
        else:
            raise ivy.exceptions.IvyException("data must be an array, list, or str")
        ivy.assertions.check_equal(
            len(ivy.shape(self._data)), 2, message="data must be 2D"
        )
        self._dtype = self._data.dtype
        self._shape = ivy.shape(self._data)

    def _process_str_data(self, data, dtype):
        is_float = "." in data
        data = data.split(";")
        for i, row in enumerate(data):
            row = row.strip().split(" ")
            data[i] = row
            for j, elem in enumerate(row):
                data[i][j] = np.float64(elem) if is_float else np.int64(elem)
        if dtype is None:
            dtype = ivy.float64 if is_float else ivy.int64
        self._data = ivy.array(data, dtype=dtype)

    # Properties #
    # ---------- #

    @property
    def A(self):
        return self._data

    @property
    def A1(self):
        return ivy.reshape(self._data, (self.size,))

    # flake8: noqa: E743, E741
    @property
    def I(self):
        if ivy.is_int_dtype(self._data):
            return ivy.inv(self._data.astype(ivy.float64))
        return ivy.inv(self._data)

    @property
    def T(self):
        return ivy.matrix_transpose(self._data)

    @property
    def data(self):
        return memoryview(ivy.to_numpy(self._data).tobytes())

    @property
    def dtype(self):
        return self._dtype

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def shape(self):
        return self._shape

    @property
    def size(self):
        return self._shape[0] * self._shape[1]

    # Setters #
    # ------- #

    @dtype.setter
    def dtype(self, dtype):
        self._data = ivy.astype(self._data, dtype)
        self._dtype = self._data.dtype

    # Built-ins #
    # --------- #

    def __repr__(self):
        sig_fig = ivy.array_significant_figures()
        dec_vals = ivy.array_decimal_values()
        rep = (
            ivy.vec_sig_fig(ivy.to_numpy(self._data), sig_fig)
            if self.size > 0
            else ivy.to_numpy(self._data)
        )
        with np.printoptions(precision=dec_vals):
            return "ivy.matrix(" + str(self._data.to_list()) + ")"

    # Instance Methods #
    # ---------------- #

    @from_zero_dim_arrays_to_float
    def argmax(self, axis=None, out=None):
        if ivy.exists(axis):
            return ivy.argmax(self.A, axis=axis, keepdims=True, out=out)
        return ivy.argmax(self.A, axis=axis, out=out)

    def any(self, axis=None, out=None):
        if ivy.exists(axis):
            return ivy.any(self.A, axis=axis, keepdims=True, out=out)
        return ivy.any(self.A, axis=axis, out=out)
