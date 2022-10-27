# global

# local
import ivy
import ivy.functional.frontends.numpy as np_frontend


class ndarray:
    def __init__(self, data):
        if ivy.is_native_array(data):
            data = ivy.Array(data)
        self.data = data

    # Instance Methods #
    # ---------------- #

    def argmax(
        self,
        /,
        *,
        axis=None,
        out=None,
        keepdims=False,
    ):
        return np_frontend.argmax(
            self.data,
            axis=axis,
            out=out,
            keepdims=keepdims,
        )

    def reshape(self, shape, order="C"):
        return np_frontend.reshape(self.data, shape)

    def transpose(self, /, axes=None):
        return np_frontend.transpose(self.data, axes=axes)

    def all(self, axis=None, out=None, keepdims=False, *, where=True):
        return np_frontend.all(self.data, axis, out, keepdims, where=where)

    def any(self, axis=None, out=None, keepdims=False, *, where=True):
        return np_frontend.any(self.data, axis, out, keepdims, where=where)

    def argsort(self, *, axis=-1, kind=None, order=None):
        return np_frontend.argsort(self.data, axis, kind, order)

    def min(self, *, axis=None, out=None, keepdims=False, initial=None, where=True):
        return np_frontend.amin(
            self.data,
            axis=axis,
            out=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
        )

    def max(self, *, axis=None, out=None, keepdims=False, initial=None, where=True):
        return np_frontend.amax(
            self.data,
            axis=axis,
            out=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
        )

    @property
    def dtype(self):
        return self.data.dtype

    def argmin(
        self,
        /,
        *,
        axis=None,
        keepdims=False,
        out=None,
    ):

        return np_frontend.argmin(
            self.data,
            axis=axis,
            keepdims=keepdims,
            out=out,
        )

    def cumprod(self, *, axis=None, dtype=None, out=None):
        return np_frontend.cumprod(
            self.data,
            axis=axis,
            dtype=dtype,
            out=out,
        )

    def cumsum(self, *, axis=None, dtype=dtype, out=None):
        return np_frontend.cumsum(
            self.data,
            axis=axis,
            dtype=dtype,
            out=out,
        )

    def sort(self, *, axis=-1, kind=None, order=None):
        return np_frontend.sort(self.data, axis=axis, kind=kind, order=order)

    def copy(self, order="C"):
        return np_frontend.copy(self.data, order=order)

    def nonzero(
        self,
    ):
        return np_frontend.nonzero(self.data)[0]

    def ravel(self, order="C"):
        return np_frontend.ravel(self.data, order=order)

    def repeat(self, repeats, axis=None):
        return np_frontend.repeat(self.data, repeats, axis=axis)

    def searchsorted(self, v, side="left", sorter=None):
        return np_frontend.searchsorted(self.data, v, side=side, sorter=sorter)

    def squeeze(self, axis=None):
        return np_frontend.squeeze(self.data, axis=axis)

    def __add__(self, value, /):
        return np_frontend.add(self.data, value)

    def __sub__(self, value, /):
        return np_frontend.subtract(self.data, value)

    def __mul__(self, value, /):
        return np_frontend.multiply(self.data, value)

    def __and__(self, value, /):
        return np_frontend.logical_and(self.data, value)

    def __or__(self, value, /):
        return np_frontend.logical_or(self.data, value)

    def __xor__(self, value, /):
        return np_frontend.logical_xor(self.data, value)

    def __matmul__(self, value, /):
        return np_frontend.matmul(self.data, value)

    def __copy__(
        self,
    ):
        return np_frontend.copy(self.data)
