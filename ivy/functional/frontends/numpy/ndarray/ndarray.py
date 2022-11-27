# global

# local
import ivy
import ivy.functional.frontends.numpy as np_frontend


class ndarray:
    def __init__(self, shape, dtype=np_frontend.float32, order=None):
        self._ivyArray = ivy.empty(shape, dtype=dtype)

        ivy.assertions.check_elem_in_list(
            order,
            ["C", "F", None],
            message="order must be one of 'C', 'F'",
        )
        if order == "F":
            self._f_contiguous = True
        else:
            self._f_contiguous = False

    # Properties #
    # ---------- #

    @property
    def ivyArray(self):
        return self._ivyArray

    @property
    def T(self):
        return np_frontend.transpose(self._ivyArray)

    @property
    def shape(self):
        return np_frontend.shape(self)

    @property
    def dtype(self):
        return self._ivyArray.dtype

    # Setters #
    # --------#

    @ivyArray.setter
    def ivyArray(self, array):
        self._ivyArray = ivy.array(array) if not isinstance(array, ivy.Array) else array

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
            self._ivyArray,
            axis=axis,
            out=out,
            keepdims=keepdims,
        )

    def reshape(self, shape, order="C"):
        ivy.assertions.check_elem_in_list(
            order,
            ["C", "F", "A", "K"],
            message="order must be one of 'C', 'F', 'A', or 'K'",
        )
        if (order in ["K", "A"] and self._f_contiguous) or order == "F":
            return np_frontend.reshape(self._ivyArray, shape, order="F")
        else:
            return np_frontend.reshape(self._ivyArray, shape, order="C")

    def transpose(self, *axes):
        if axes and isinstance(axes[0], tuple):
            axes = axes[0]
        return np_frontend.transpose(self._ivyArray, axes=axes)

    def swapaxes(self, axis1, axis2, /):
        return np_frontend.swapaxes(self._ivyArray, axis1, axis2)

    def all(self, axis=None, out=None, keepdims=False, *, where=True):
        return np_frontend.all(self._ivyArray, axis, out, keepdims, where=where)

    def any(self, axis=None, out=None, keepdims=False, *, where=True):
        return np_frontend.any(self._ivyArray, axis, out, keepdims, where=where)

    def argsort(self, *, axis=-1, kind=None, order=None):
        return np_frontend.argsort(self._ivyArray, axis, kind, order)

    def mean(self, *, axis=None, dtype=None, out=None, keepdims=False, where=True):
        return np_frontend.mean(
            self._ivyArray,
            axis=axis,
            dtype=dtype,
            out=out,
            keepdims=keepdims,
            where=where,
        )

    def min(self, *, axis=None, out=None, keepdims=False, initial=None, where=True):
        return np_frontend.amin(
            self._ivyArray,
            axis=axis,
            out=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
        )

    def max(self, *, axis=None, out=None, keepdims=False, initial=None, where=True):
        return np_frontend.amax(
            self._ivyArray,
            axis=axis,
            out=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
        )

    def argmin(
        self,
        /,
        *,
        axis=None,
        keepdims=False,
        out=None,
    ):

        return np_frontend.argmin(
            self._ivyArray,
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
            self._ivyArray,
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
            self._ivyArray,
            axis=axis,
            dtype=dtype,
            out=out,
        )

    def cumsum(self, *, axis=None, dtype=None, out=None):
        return np_frontend.cumsum(
            self._ivyArray,
            axis=axis,
            dtype=dtype,
            out=out,
        )

    def sort(self, *, axis=-1, kind=None, order=None):
        return np_frontend.sort(self._ivyArray, axis=axis, kind=kind, order=order)

    def copy(self, order="C"):
        return np_frontend.copy(self._ivyArray, order=order)

    def nonzero(
        self,
    ):
        return np_frontend.nonzero(self._ivyArray)[0]

    def ravel(self, order="C"):
        ivy.assertions.check_elem_in_list(
            order,
            ["C", "F", "A", "K"],
            message="order must be one of 'C', 'F', 'A', or 'K'",
        )
        if (order in ["K", "A"] and self._f_contiguous) or order == "F":
            return np_frontend.ravel(self._ivyArray, order="F")
        else:
            return np_frontend.ravel(self._ivyArray, order="C")

    def flatten(self, order="C"):
        ivy.assertions.check_elem_in_list(
            order,
            ["C", "F", "A", "K"],
            message="order must be one of 'C', 'F', 'A', or 'K'",
        )
        if (order in ["K", "A"] and self._f_contiguous) or order == "F":
            return np_frontend.ravel(self._ivyArray, order="F")
        else:
            return np_frontend.ravel(self._ivyArray, order="C")

    def repeat(self, repeats, axis=None):
        return np_frontend.repeat(self._ivyArray, repeats, axis=axis)

    def searchsorted(self, v, side="left", sorter=None):
        return np_frontend.searchsorted(self._ivyArray, v, side=side, sorter=sorter)

    def squeeze(self, axis=None):
        return np_frontend.squeeze(self._ivyArray, axis=axis)

    def std(
        self, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True
    ):
        return np_frontend.std(
            self._ivyArray,
            axis=axis,
            dtype=dtype,
            out=out,
            ddof=ddof,
            keepdims=keepdims,
            where=where,
        )

    def __add__(self, value, /):
        return np_frontend.add(self._ivyArray, value)

    def __sub__(self, value, /):
        return np_frontend.subtract(self._ivyArray, value)

    def __mul__(self, value, /):
        return np_frontend.multiply(self._ivyArray, value)

    def __truediv__(self, value, /):
        return np_frontend.true_divide(self._ivyArray, value)

    def __and__(self, value, /):
        return np_frontend.logical_and(self._ivyArray, value)

    def __or__(self, value, /):
        return np_frontend.logical_or(self._ivyArray, value)

    def __xor__(self, value, /):
        return np_frontend.logical_xor(self._ivyArray, value)

    def __matmul__(self, value, /):
        return np_frontend.matmul(self._ivyArray, value)

    def __copy__(
        self,
    ):
        return np_frontend.copy(self._ivyArray)

    def __neg__(
        self,
    ):
        return np_frontend.negative(self._ivyArray)

    def __pos__(
        self,
    ):
        return np_frontend.positive(self._ivyArray)

    def __bool__(
        self,
    ):
        if isinstance(self._ivyArray, int):
            return self._ivyArray != 0

        temp = ivy.squeeze(ivy.asarray(self._ivyArray), axis=None)
        shape = ivy.shape(temp)
        if shape:
            raise ValueError(
                "The truth value of an array with more than one element is ambiguous. "
                "Use a.any() or a.all()"
            )

        return temp != 0

    def __ne__(self, value, /):
        return np_frontend.not_equal(self._ivyArray, value)

    def __eq__(self, value, /):
        return ivy.array(np_frontend.equal(self._ivyArray, value), dtype=ivy.bool)

    def __ge__(self, value, /):
        return np_frontend.greater_equal(self._ivyArray, value)

    def __gt__(self, value, /):
        return np_frontend.greater(self._ivyArray, value)

    def __le__(self, value, /):
        return np_frontend.less_equal(self._ivyArray, value)

    def __lt__(self, value, /):
        return np_frontend.less(self._ivyArray, value)

    def __int__(
        self,
    ):
        return ivy.array(ivy.reshape(self._ivyArray, -1), dtype=ivy.int64)[0]

    def __float__(
        self,
    ):
        return ivy.array(ivy.reshape(self._ivyArray, -1), dtype=ivy.float64)[0]

    def __contains__(self, key, /):
        return key in ivy.reshape(self._ivyArray, -1)

    def __iadd__(self, value, /):
        return np_frontend.add(self._ivyArray, value)

    def __isub__(self, value, /):
        return np_frontend.subtract(self._ivyArray, value)

    def __imul__(self, value, /):
        return np_frontend.multiply(self._ivyArray, value)

    def __ipow__(self, value, /):
        return np_frontend.power(self._ivyArray, value)

    def __iand__(self, value, /):
        return np_frontend.logical_and(self._ivyArray, value)

    def __ior__(self, value, /):
        return np_frontend.logical_or(self._ivyArray, value)

    def __ixor__(self, value, /):
        return np_frontend.logical_xor(self._ivyArray, value)

    def __imod__(self, value, /):
        return np_frontend.mod(self._ivyArray, value)

    def __abs__(self):
        return np_frontend.absolute(self._ivyArray)
