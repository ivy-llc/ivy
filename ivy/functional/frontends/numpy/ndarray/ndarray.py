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

    def transpose(self, *axes):
        if axes and isinstance(axes[0], tuple):
            axes = axes[0]
        return np_frontend.transpose(self.data, axes=axes)

    @property
    def T(self):
        return np_frontend.transpose(self.data)

    @property
    def shape(self):
        return np_frontend.shape(self)

    def swapaxes(self, axis1, axis2, /):
        return np_frontend.swapaxes(self.data, axis1, axis2)

    def all(self, axis=None, out=None, keepdims=False, *, where=True):
        return np_frontend.all(self.data, axis, out, keepdims, where=where)

    def any(self, axis=None, out=None, keepdims=False, *, where=True):
        return np_frontend.any(self.data, axis, out, keepdims, where=where)

    def argsort(self, *, axis=-1, kind=None, order=None):
        return np_frontend.argsort(self.data, axis, kind, order)

    def mean(self, *, axis=None, dtype=None, out=None, keepdims=False, where=True):
        return np_frontend.mean(
            self.data, axis=axis, dtype=dtype, out=out, keepdims=keepdims, where=where
        )

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

    def clip(
        self,
        a_min,
        a_max,
        /,
        out=None,
        *,
        where=True,
        casting="same_kind",
        order="k",
        dtype=None,
        subok=True,
    ):
        return np_frontend.clip(
            self.data,
            a_min,
            a_max,
            out=out,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
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

    def __truediv__(self, value, /):
        return np_frontend.true_divide(self.data, value)

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

    def __neg__(
        self,
    ):
        return np_frontend.negative(self.data)

    def __pos__(
        self,
    ):
        return np_frontend.positive(self.data)

    def __bool__(
        self,
    ):
        if isinstance(self.data, int):
            return self.data != 0

        temp = ivy.squeeze(ivy.asarray(self.data), axis=None)
        shape = ivy.shape(temp)
        if shape:
            raise ValueError(
                "The truth value of an array with more than one element is ambiguous. "
                "Use a.any() or a.all()"
            )

        return temp != 0

    def __ne__(self, value, /):
        return np_frontend.not_equal(self.data, value)

    def __eq__(self, value, /):
        return ivy.array(np_frontend.equal(self.data, value), dtype=ivy.bool)

    def __ge__(self, value, /):
        return np_frontend.greater_equal(self.data, value)

    def __gt__(self, value, /):
        return np_frontend.greater(self.data, value)

    def __le__(self, value, /):
        return np_frontend.less_equal(self.data, value)

    def __lt__(self, value, /):
        return np_frontend.less(self.data, value)

    def __int__(
        self,
    ):
        return ivy.array(ivy.reshape(self.data, -1), dtype=ivy.int64)[0]

    def __float__(
        self,
    ):
        return ivy.array(ivy.reshape(self.data, -1), dtype=ivy.float64)[0]

    def __contains__(self, key, /):
        return key in ivy.reshape(self.data, -1)

    def __iadd__(self, value, /):
        return np_frontend.add(self.data, value)

    def __isub__(self, value, /):
        return np_frontend.subtract(self.data, value)

    def __imul__(self, value, /):
        return np_frontend.multiply(self.data, value)

    def __ipow__(self, value, /):
        return np_frontend.power(self.data, value)

    def __iand__(self, value, /):
        return np_frontend.logical_and(self.data, value)

    def __ior__(self, value, /):
        return np_frontend.logical_or(self.data, value)

    def __ixor__(self, value, /):
        return np_frontend.logical_xor(self.data, value)

    def __imod__(self, value, /):
        return np_frontend.mod(self.data, value)
