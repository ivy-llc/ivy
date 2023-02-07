# global

# local
import ivy
import ivy.functional.frontends.numpy as np_frontend


class ndarray:
    def __init__(self, shape, dtype="float32", order=None):
        if isinstance(dtype, np_frontend.dtype):
            dtype = dtype.ivy_dtype
        self._ivy_array = ivy.empty(shape, dtype=dtype)

        ivy.assertions.check_elem_in_list(
            order,
            ["C", "F", None],
            message="order must be one of 'C', 'F'",
        )
        if order == "F":
            self._f_contiguous = True
        else:
            self._f_contiguous = False

    def __repr__(self):
        return str(self._ivy_array.__repr__()).replace(
            "ivy.array", "ivy.frontends.numpy.ndarray"
        )

    # Properties #
    # ---------- #

    @property
    def ivy_array(self):
        return self._ivy_array

    @property
    def T(self):
        return np_frontend.transpose(self._ivy_array)

    @property
    def shape(self):
        return np_frontend.shape(self)

    @property
    def dtype(self):
        return np_frontend.dtype(self._ivy_array.dtype)

    # Setters #
    # --------#

    @ivy_array.setter
    def ivy_array(self, array):
        self._ivy_array = (
            ivy.array(array) if not isinstance(array, ivy.Array) else array
        )

    # Instance Methods #
    # ---------------- #

    def astype(self, dtype, order="K", casting="unsafe", subok=True, copy=True):
        ivy.assertions.check_elem_in_list(
            order,
            ["C", "F", "A", "K"],
            message="order must be one of 'C', 'F', or 'A'",
        )
        if copy and self._f_contiguous:
            ret = np_frontend.array(self._ivy_array, order="F")
        else:
            ret = np_frontend.array(self._ivy_array) if copy else self

        dtype = np_frontend.to_ivy_dtype(dtype)
        if np_frontend.can_cast(ret._ivy_array, dtype, casting=casting):
            ret._ivy_array = ret._ivy_array.astype(dtype)
        else:
            raise ivy.exceptions.IvyException(
                f"Cannot cast array data from dtype('{ret._ivy_array.dtype}')"
                f" to dtype('{dtype}') according to the rule '{casting}'"
            )
        if order == "F":
            ret._f_contiguous = True
        elif order == "C":
            ret._f_contiguous = False
        return ret

    def argmax(
        self,
        /,
        *,
        axis=None,
        out=None,
        keepdims=False,
    ):
        return np_frontend.argmax(
            self._ivy_array,
            axis=axis,
            out=out,
            keepdims=keepdims,
        )

    def reshape(self, newshape, /, *, order="C"):
        ivy.assertions.check_elem_in_list(
            order,
            ["C", "F", "A"],
            message="order must be one of 'C', 'F', or 'A'",
        )
        if (order == "A" and self._f_contiguous) or order == "F":
            return np_frontend.reshape(self._ivy_array, newshape, order="F")
        else:
            return np_frontend.reshape(self._ivy_array, newshape, order="C")

    def transpose(self, axes, /):
        if axes and isinstance(axes[0], tuple):
            axes = axes[0]
        return np_frontend.transpose(self._ivy_array, axes=axes)

    def swapaxes(self, axis1, axis2, /):
        return np_frontend.swapaxes(self._ivy_array, axis1, axis2)

    def all(self, axis=None, out=None, keepdims=False, *, where=True):
        return np_frontend.all(self._ivy_array, axis, out, keepdims, where=where)

    def any(self, axis=None, out=None, keepdims=False, *, where=True):
        return np_frontend.any(self._ivy_array, axis, out, keepdims, where=where)

    def argsort(self, *, axis=-1, kind=None, order=None):
        return np_frontend.argsort(self._ivy_array, axis=axis, kind=kind, order=order)

    def mean(self, *, axis=None, dtype=None, out=None, keepdims=False, where=True):
        return np_frontend.mean(
            self._ivy_array,
            axis=axis,
            dtype=dtype,
            out=out,
            keepdims=keepdims,
            where=where,
        )

    def min(self, *, axis=None, out=None, keepdims=False, initial=None, where=True):
        return np_frontend.amin(
            self._ivy_array,
            axis=axis,
            out=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
        )

    def max(self, *, axis=None, out=None, keepdims=False, initial=None, where=True):
        return np_frontend.amax(
            self._ivy_array,
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
            self._ivy_array,
            axis=axis,
            keepdims=keepdims,
            out=out,
        )

    def clip(
        self,
        min,
        max,
        /,
        out=None,
        *,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
    ):
        return np_frontend.clip(
            self._ivy_array,
            min,
            max,
            out=out,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
        )

    def cumprod(self, *, axis=None, dtype=None, out=None):
        return np_frontend.cumprod(
            self._ivy_array,
            axis=axis,
            dtype=dtype,
            out=out,
        )

    def cumsum(self, *, axis=None, dtype=None, out=None):
        return np_frontend.cumsum(
            self._ivy_array,
            axis=axis,
            dtype=dtype,
            out=out,
        )

    def diagonal(self, *, offset=0, axis1=0, axis2=1):
        return np_frontend.diagonal(
            self._ivyArray,
            offset=offset,
            axis1=axis1,
            axis2=axis2,
        )

    def sort(self, *, axis=-1, kind=None, order=None):
        return np_frontend.sort(self._ivy_array, axis=axis, kind=kind, order=order)

    def copy(self, order="C"):
        return np_frontend.copy(self._ivy_array, order=order)

    def nonzero(
        self,
    ):
        return np_frontend.nonzero(self._ivy_array)[0]

    def ravel(self, order="C"):
        ivy.assertions.check_elem_in_list(
            order,
            ["C", "F", "A", "K"],
            message="order must be one of 'C', 'F', 'A', or 'K'",
        )
        if (order in ["K", "A"] and self._f_contiguous) or order == "F":
            return np_frontend.ravel(self._ivy_array, order="F")
        else:
            return np_frontend.ravel(self._ivy_array, order="C")

    def flatten(self, order="C"):
        ivy.assertions.check_elem_in_list(
            order,
            ["C", "F", "A", "K"],
            message="order must be one of 'C', 'F', 'A', or 'K'",
        )
        if (order in ["K", "A"] and self._f_contiguous) or order == "F":
            return np_frontend.ravel(self._ivy_array, order="F")
        else:
            return np_frontend.ravel(self._ivy_array, order="C")

    def repeat(self, repeats, axis=None):
        return np_frontend.repeat(self._ivy_array, repeats, axis=axis)

    def searchsorted(self, v, side="left", sorter=None):
        return np_frontend.searchsorted(self._ivy_array, v, side=side, sorter=sorter)

    def squeeze(self, axis=None):
        return np_frontend.squeeze(self._ivy_array, axis=axis)

    def std(
        self, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True
    ):
        return np_frontend.std(
            self._ivy_array,
            axis=axis,
            dtype=dtype,
            out=out,
            ddof=ddof,
            keepdims=keepdims,
            where=where,
        )

    def __add__(self, value, /):
        return np_frontend.add(self._ivy_array, value)

    def __radd__(self, value, /):
        return np_frontend.add(self._ivy_array, value)

    def __sub__(self, value, /):
        return np_frontend.subtract(self._ivy_array, value)

    def __mul__(self, value, /):
        return np_frontend.multiply(self._ivy_array, value)

    def __truediv__(self, value, /):
        return np_frontend.true_divide(self._ivy_array, value)

    def __floordiv__(self, value, /):
        return np_frontend.floor_divide(self._ivy_array, value)

    def __rtruediv__(self, value, /):
        return np_frontend.true_divide(value, self._ivy_array)

    def __pow__(self, value, /):
        return np_frontend.power(self._ivy_array, value)

    def __and__(self, value, /):
        return np_frontend.logical_and(self._ivy_array, value)

    def __or__(self, value, /):
        return np_frontend.logical_or(self._ivy_array, value)

    def __xor__(self, value, /):
        return np_frontend.logical_xor(self._ivy_array, value)

    def __matmul__(self, value, /):
        return np_frontend.matmul(self._ivy_array, value)

    def __copy__(
        self,
    ):
        return np_frontend.copy(self._ivy_array)

    def __neg__(
        self,
    ):
        return np_frontend.negative(self._ivy_array)

    def __pos__(
        self,
    ):
        return np_frontend.positive(self._ivy_array)

    def __bool__(
        self,
    ):
        if isinstance(self._ivy_array, int):
            return self._ivy_array != 0

        temp = ivy.squeeze(ivy.asarray(self._ivy_array), axis=None)
        shape = ivy.shape(temp)
        if shape:
            raise ValueError(
                "The truth value of an array with more than one element is ambiguous. "
                "Use a.any() or a.all()"
            )

        return temp != 0

    def __ne__(self, value, /):
        return np_frontend.not_equal(self._ivy_array, value)

    def __len__(self):
        return len(self.ivy_array)

    def __eq__(self, value, /):
        return ivy.array(np_frontend.equal(self._ivy_array, value), dtype=ivy.bool)

    def __ge__(self, value, /):
        return np_frontend.greater_equal(self._ivy_array, value)

    def __gt__(self, value, /):
        return np_frontend.greater(self._ivy_array, value)

    def __le__(self, value, /):
        return np_frontend.less_equal(self._ivy_array, value)

    def __lt__(self, value, /):
        return np_frontend.less(self._ivy_array, value)

    def __int__(
        self,
    ):
        return ivy.array(ivy.reshape(self._ivy_array, -1), dtype=ivy.int64)[0]

    def __float__(
        self,
    ):
        return ivy.array(ivy.reshape(self._ivy_array, -1), dtype=ivy.float64)[0]

    def __contains__(self, key, /):
        return key in ivy.reshape(self._ivy_array, -1)

    def __iadd__(self, value, /):
        return np_frontend.add(self._ivy_array, value)

    def __isub__(self, value, /):
        return np_frontend.subtract(self._ivy_array, value)

    def __imul__(self, value, /):
        return np_frontend.multiply(self._ivy_array, value)

    def __itruediv__(self, value, /):
        return np_frontend.true_divide(self._ivy_array, value)

    def __ipow__(self, value, /):
        return np_frontend.power(self._ivy_array, value)

    def __iand__(self, value, /):
        return np_frontend.logical_and(self._ivy_array, value)

    def __ior__(self, value, /):
        return np_frontend.logical_or(self._ivy_array, value)

    def __ixor__(self, value, /):
        return np_frontend.logical_xor(self._ivy_array, value)

    def __imod__(self, value, /):
        return np_frontend.mod(self._ivy_array, value)

    def __abs__(self):
        return np_frontend.absolute(self._ivy_array)

    def __getitem__(self, query):
        ret = ivy.get_item(self._ivy_array, query)
        return np_frontend.numpy_dtype_to_scalar[ivy.dtype(self._ivy_array)](
            ivy.array(ret, dtype=ivy.dtype(ret), copy=False)
        )

    def __setitem__(self, key, value):
        if hasattr(value, "ivy_array"):
            value = (
                ivy.to_scalar(value.ivy_array)
                if value.shape == ()
                else ivy.to_list(value)
            )
        self._ivy_array[key] = value
