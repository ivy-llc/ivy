# global
import ivy

# local
from ivy.func_wrapper import from_zero_dim_arrays_to_float


class matrix:
    def __init__(self, data, dtype=None, copy=True):
        self._init_data(data, dtype)

    def _init_data(self, data, dtype):
        if isinstance(data, str):
            self._process_str_data(data, dtype)
        elif isinstance(data, list) or ivy.is_array(data):
            data = (
                ivy.array(data, dtype=dtype) if ivy.exists(dtype) else ivy.array(data)
            )
            ivy.assertions.check_equal(len(ivy.shape(data)), 2)
            self._data = data
        else:
            raise ivy.exceptions.IvyException("data must be a 2D array, list, or str")
        self._shape = ivy.shape(self._data)
        self._dtype = self._data.dtype

    def _process_str_data(self, data, dtype):
        is_float = "." in data
        data = data.split(";")
        ivy.assertions.check_equal(
            len(data), 2, message="only one semicolon should exist for rows splitting"
        )
        for i in range(2):
            data[i] = data[i].split(",") if "," in data[i] else data[i].split()
            data[i] = [
                float(x.strip()) if is_float else int(x.strip()) for x in data[i]
            ]
        ivy.assertions.check_equal(
            len(data[0]), len(data[1]), message="elements in each row is unequal"
        )
        self._data = (
            ivy.array(data, dtype=dtype) if ivy.exists(dtype) else ivy.array(data)
        )

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
        return ivy.inv(self._data)

    @property
    def T(self):
        return ivy.matrix_transpose(self._data)

    @property
    def data(self):
        return hex(id(self._data))

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

    # TODO: check setters

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
