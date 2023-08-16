# global
import struct
import warnings

# local
import ivy
import ivy.functional.frontends.numpy as np_frontend
from ivy.functional.frontends.numpy.func_wrapper import _to_ivy_array


# --- Helpers ---#
# ---------------#


# tobytes helper function
def _to_bytes_helper(array, order="C"):
    def _integers_bytes_repr(item_val, /, *, dtype=None):
        if dtype == ivy.int8:
            return item_val.to_bytes(1, byteorder="big", signed=True)
        elif dtype == ivy.int16:
            return struct.pack("h", item_val)
        elif dtype == ivy.int32:
            return struct.pack("i", item_val)
        elif dtype == ivy.int64:
            return struct.pack("q", item_val)

    def _float_bytes_repr(item_val, /, *, dtype=None):
        if dtype == ivy.float16:
            return struct.pack("e", item_val)
        elif dtype == ivy.float32:
            return struct.pack("f", item_val)
        return struct.pack("d", item_val)

    def _bool_bytes_repr(item_val, /):
        return struct.pack("?", item_val)

    def _complex_bytes_repr(item_val, /, *, dtype=None):
        if dtype == ivy.complex64:
            # complex64 is represented as two 32-bit floats
            return struct.pack("ff", item_val.real, item_val.imag)

        elif dtype == ivy.complex128:
            # complex128 is represented as two 64-bit floats
            return struct.pack("dd", item_val.real, item_val.imag)

    def _unsigned_int_bytes_repr(item_val, /, *, dtype=None):
        if dtype == ivy.uint8:
            return item_val.to_bytes(1, byteorder="little", signed=False)
        elif dtype == ivy.uint16:
            return struct.pack("H", item_val)
        elif dtype == ivy.uint32:
            return struct.pack("I", item_val)
        elif dtype == ivy.uint64:
            return struct.pack("Q", item_val)

    if ivy.get_num_dims(array) == 0:
        scalar_value = ivy.to_scalar(array)
        dtype = ivy.dtype(array)
        if ivy.is_int_dtype(dtype) and not ivy.is_uint_dtype(dtype):
            return _integers_bytes_repr(scalar_value, dtype=dtype)

        elif ivy.is_float_dtype(dtype):
            return _float_bytes_repr(scalar_value, dtype=dtype)

        elif ivy.is_bool_dtype(dtype):
            return _bool_bytes_repr(scalar_value)

        elif ivy.is_complex_dtype(dtype):
            return _complex_bytes_repr(scalar_value, dtype=dtype)

        elif ivy.is_uint_dtype(dtype):
            return _unsigned_int_bytes_repr(scalar_value, dtype=dtype)
        else:
            raise ValueError("Unsupported data type for the array.")
    else:
        if order == "F":
            array = np_frontend.ravel(array, order="F").ivy_array
        array = ivy.flatten(array)
        if ivy.is_int_dtype(array) and not ivy.is_uint_dtype(array):
            bytes_reprs = [
                _integers_bytes_repr(item, dtype=ivy.dtype(array))
                for item in array.to_list()
            ]
            return b"".join(bytes_reprs)

        elif ivy.is_float_dtype(array):
            bytes_reprs = [
                _float_bytes_repr(item, dtype=ivy.dtype(array))
                for item in array.to_list()
            ]
            return b"".join(bytes_reprs)

        elif ivy.is_bool_dtype(array):
            bytes_reprs = [_bool_bytes_repr(item) for item in array.to_list()]
            return b"".join(bytes_reprs)

        elif ivy.is_complex_dtype(array):
            bytes_reprs = [
                _complex_bytes_repr(item, dtype=ivy.dtype(array))
                for item in array.to_list()
            ]
            return b"".join(bytes_reprs)

        elif ivy.is_uint_dtype(array):
            bytes_reprs = [
                _unsigned_int_bytes_repr(item, dtype=ivy.dtype(array))
                for item in array.to_list()
            ]
            return b"".join(bytes_reprs)
        else:
            raise ValueError("Unsupported data type for the array.")


# --- Classes ---#
# ---------------#


class ndarray:
    def __init__(self, shape, dtype="float32", order=None, _init_overload=False):
        if isinstance(dtype, np_frontend.dtype):
            dtype = dtype.ivy_dtype

        # in thise case shape is actually the desired array
        if _init_overload:
            self._ivy_array = (
                ivy.array(shape) if not isinstance(shape, ivy.Array) else shape
            )
        else:
            self._ivy_array = ivy.empty(shape=shape, dtype=dtype)

        ivy.utils.assertions.check_elem_in_list(
            order,
            ["C", "F", None],
            message="order must be one of 'C', 'F'",
        )
        if order == "F":
            self._f_contiguous = True
        else:
            self._f_contiguous = False

    def __repr__(self):
        return str(self.ivy_array.__repr__()).replace(
            "ivy.array", "ivy.frontends.numpy.ndarray"
        )

    # Properties #
    # ---------- #

    @property
    def ivy_array(self):
        return self._ivy_array

    @property
    def T(self):
        return np_frontend.transpose(self)

    @property
    def shape(self):
        return self.ivy_array.shape

    @property
    def size(self):
        return self.ivy_array.size

    @property
    def dtype(self):
        return np_frontend.dtype(self.ivy_array.dtype)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def flat(self):
        self = self.flatten()
        return self

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
        ivy.utils.assertions.check_elem_in_list(
            order,
            ["C", "F", "A", "K"],
            message="order must be one of 'C', 'F', or 'A'",
        )
        if copy and self._f_contiguous:
            ret = np_frontend.array(self.ivy_array, order="F")
        else:
            ret = np_frontend.array(self.ivy_array) if copy else self

        dtype = np_frontend.to_ivy_dtype(dtype)
        if np_frontend.can_cast(ret, dtype, casting=casting):
            ret.ivy_array = ret.ivy_array.astype(dtype)
        else:
            raise ivy.utils.exceptions.IvyException(
                f"Cannot cast array data from dtype('{ret.ivy_array.dtype}')"
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
            self,
            axis=axis,
            out=out,
            keepdims=keepdims,
        )

    def reshape(self, newshape, /, *, order="C"):
        ivy.utils.assertions.check_elem_in_list(
            order,
            ["C", "F", "A"],
            message="order must be one of 'C', 'F', or 'A'",
        )
        if (order == "A" and self._f_contiguous) or order == "F":
            return np_frontend.reshape(self, newshape, order="F")
        else:
            return np_frontend.reshape(self, newshape, order="C")

    def resize(self, newshape, /, *, refcheck=True):
        return np_frontend.resize(self, newshape, refcheck)

    def transpose(self, axes, /):
        if axes and isinstance(axes[0], tuple):
            axes = axes[0]
        return np_frontend.transpose(self, axes=axes)

    def swapaxes(self, axis1, axis2, /):
        return np_frontend.swapaxes(self, axis1, axis2)

    def all(self, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
        if not (dtype is None or ivy.is_bool_dtype(dtype)):
            raise TypeError(
                "No loop matching the specified signature and "
                "casting was found for ufunc logical_or"
            )
        return np_frontend.all(self, axis, out, keepdims, where=where)

    def any(self, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
        if not (dtype is None or ivy.is_bool_dtype(dtype)):
            raise TypeError(
                "No loop matching the specified signature and "
                "casting was found for ufunc logical_or"
            )
        return np_frontend.any(self, axis, out, keepdims, where=where)

    def argsort(self, *, axis=-1, kind=None, order=None):
        return np_frontend.argsort(self, axis=axis, kind=kind, order=order)

    def mean(self, *, axis=None, dtype=None, out=None, keepdims=False, where=True):
        return np_frontend.mean(
            self,
            axis=axis,
            dtype=dtype,
            out=out,
            keepdims=keepdims,
            where=where,
        )

    def min(self, *, axis=None, out=None, keepdims=False, initial=None, where=True):
        return np_frontend.amin(
            self,
            axis=axis,
            out=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
        )

    def max(self, *, axis=None, out=None, keepdims=False, initial=None, where=True):
        return np_frontend.amax(
            self,
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
            self,
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
            self,
            min,
            max,
            out=out,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
        )

    def compress(self, condition, axis=None, out=None):
        return np_frontend.compress(
            condition=condition,
            a=self,
            axis=axis,
            out=out,
        )

    def conj(
        self,
        /,
        out=None,
        *,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
    ):
        return np_frontend.conj(
            self.ivy_array,
            out=out,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
        )

    def cumprod(self, *, axis=None, dtype=None, out=None):
        return np_frontend.cumprod(
            self,
            axis=axis,
            dtype=dtype,
            out=out,
        )

    def cumsum(self, *, axis=None, dtype=None, out=None):
        return np_frontend.cumsum(
            self,
            axis=axis,
            dtype=dtype,
            out=out,
        )

    def dot(self, b, out=None):
        return np_frontend.dot(self, b, out=out)

    def diagonal(self, *, offset=0, axis1=0, axis2=1):
        return np_frontend.diagonal(
            self,
            offset=offset,
            axis1=axis1,
            axis2=axis2,
        )

    def sort(self, *, axis=-1, kind=None, order=None):
        return np_frontend.sort(self, axis=axis, kind=kind, order=order)

    def copy(self, order="C"):
        return np_frontend.copy(self, order=order)

    def nonzero(
        self,
    ):
        return np_frontend.nonzero(self)[0]

    def ravel(self, order="C"):
        ivy.utils.assertions.check_elem_in_list(
            order,
            ["C", "F", "A", "K"],
            message="order must be one of 'C', 'F', 'A', or 'K'",
        )
        if (order in ["K", "A"] and self._f_contiguous) or order == "F":
            return np_frontend.ravel(self, order="F")
        else:
            return np_frontend.ravel(self, order="C")

    def flatten(self, order="C"):
        ivy.utils.assertions.check_elem_in_list(
            order,
            ["C", "F", "A", "K"],
            message="order must be one of 'C', 'F', 'A', or 'K'",
        )
        if (order in ["K", "A"] and self._f_contiguous) or order == "F":
            return np_frontend.ravel(self, order="F")
        else:
            return np_frontend.ravel(self, order="C")

    def fill(self, num, /):
        self.ivy_array = np_frontend.full(self.shape, num).ivy_array
        return

    def repeat(self, repeats, axis=None):
        return np_frontend.repeat(self, repeats, axis=axis)

    def searchsorted(self, v, side="left", sorter=None):
        return np_frontend.searchsorted(self, v, side=side, sorter=sorter)

    def squeeze(self, axis=None):
        return np_frontend.squeeze(self, axis=axis)

    def std(
        self, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True
    ):
        return np_frontend.std(
            self,
            axis=axis,
            dtype=dtype,
            out=out,
            ddof=ddof,
            keepdims=keepdims,
            where=where,
        )

    def tobytes(self, order="C"):
        return _to_bytes_helper(self.ivy_array, order=order)

    def tostring(self, order="C"):
        warnings.warn(
            "DeprecationWarning: tostring() is deprecated. Use tobytes() instead."
        )
        return self.tobytes(order=order)

    def prod(
        self,
        *,
        axis=None,
        dtype=None,
        out=None,
        keepdims=False,
        initial=None,
        where=True,
    ):
        return np_frontend.prod(
            self,
            axis=axis,
            dtype=dtype,
            keepdims=keepdims,
            initial=initial,
            where=where,
            out=out,
        )

    def tofile(self, fid, /, sep="", format_="%s"):
        if self.ndim == 0:
            string = str(self)
        else:
            string = sep.join([str(item) for item in self.tolist()])
        with open(fid, "w") as f:
            f.write(string)

    def tolist(self) -> list:
        return self._ivy_array.to_list()

    def view(self):
        return np_frontend.reshape(self, tuple(self.shape))

    def __add__(self, value, /):
        return np_frontend.add(self, value)

    def __radd__(self, value, /):
        return np_frontend.add(self, value)

    def __sub__(self, value, /):
        return np_frontend.subtract(self, value)

    def __mul__(self, value, /):
        return np_frontend.multiply(self, value)

    def __rmul__(self, value, /):
        return np_frontend.multiply(value, self)

    def __truediv__(self, value, /):
        return np_frontend.true_divide(self, value)

    def __floordiv__(self, value, /):
        return np_frontend.floor_divide(self, value)

    def __rtruediv__(self, value, /):
        return np_frontend.true_divide(value, self)

    def __pow__(self, value, /):
        return np_frontend.power(self, value)

    def __and__(self, value, /):
        return np_frontend.logical_and(self, value)

    def __or__(self, value, /):
        return np_frontend.logical_or(self, value)

    def __xor__(self, value, /):
        return np_frontend.logical_xor(self, value)

    def __matmul__(self, value, /):
        return np_frontend.matmul(self, value)

    def __copy__(
        self,
    ):
        return np_frontend.copy(self)

    def __deepcopy__(self, memo, /):
        return self.ivy_array.__deepcopy__(memo)

    def __neg__(
        self,
    ):
        return np_frontend.negative(self)

    def __pos__(
        self,
    ):
        return np_frontend.positive(self)

    def __bool__(
        self,
    ):
        if isinstance(self.ivy_array, int):
            return self.ivy_array != 0

        temp = ivy.squeeze(ivy.asarray(self.ivy_array), axis=None)
        if ivy.get_num_dims(temp) > 1:
            raise ValueError(
                "The truth value of an array with more than one element is ambiguous. "
                "Use a.any() or a.all()"
            )

        return temp != 0

    def __ne__(self, value, /):
        return np_frontend.not_equal(self, value)

    def __len__(self):
        return len(self.ivy_array)

    def __eq__(self, value, /):
        return np_frontend.equal(self, value)

    def __ge__(self, value, /):
        return np_frontend.greater_equal(self, value)

    def __gt__(self, value, /):
        return np_frontend.greater(self, value)

    def __le__(self, value, /):
        return np_frontend.less_equal(self, value)

    def __lt__(self, value, /):
        return np_frontend.less(self, value)

    def __int__(
        self,
    ):
        if "complex" in self.dtype.name:
            raise TypeError(
                "int() argument must be a string, a bytes-like object or a number, not"
                " 'complex"
            )
        return int(self.ivy_array)

    def __float__(
        self,
    ):
        if "complex" in self.dtype.name:
            raise TypeError(
                "float() argument must be a string or a real number, not 'complex"
            )
        return float(self.ivy_array)

    def __complex__(
        self,
    ):
        return complex(self.ivy_array)

    def __contains__(self, key, /):
        return np_frontend.any(self == key)

    def __iadd__(self, value, /):
        return np_frontend.add(self, value, out=self)

    def __isub__(self, value, /):
        return np_frontend.subtract(self, value, out=self)

    def __imul__(self, value, /):
        return np_frontend.multiply(self, value, out=self)

    def __itruediv__(self, value, /):
        return np_frontend.true_divide(self, value, out=self)

    def __ifloordiv__(self, value, /):
        return np_frontend.floor_divide(self, value, out=self)

    def __ipow__(self, value, /):
        return np_frontend.power(self, value, out=self)

    def __iand__(self, value, /):
        return np_frontend.logical_and(self, value, out=self)

    def __ior__(self, value, /):
        return np_frontend.logical_or(self, value, out=self)

    def __ixor__(self, value, /):
        return np_frontend.logical_xor(self, value, out=self)

    def __imod__(self, value, /):
        return np_frontend.mod(self, value, out=self)

    def __invert__(self, /):
        return ivy.bitwise_invert(self.ivy_array)

    def __abs__(self):
        return np_frontend.absolute(self)

    def __array__(self, dtype=None, /):
        if not dtype:
            return self
        return np_frontend.array(self, dtype=dtype)

    def __array_wrap__(self, array, context=None, /):
        if context is None:
            return np_frontend.array(array)
        else:
            return np_frontend.asarray(self)

    def __getitem__(self, key, /):
        ivy_args = ivy.nested_map([self, key], _to_ivy_array)
        ret = ivy.get_item(*ivy_args)
        return np_frontend.ndarray(ret, _init_overload=True)

    def __setitem__(self, key, value, /):
        key, value = ivy.nested_map([key, value], _to_ivy_array)
        self.ivy_array[key] = value

    def __iter__(self):
        if self.ndim == 0:
            raise TypeError("iteration over a 0-d ndarray not supported")
        for i in range(self.shape[0]):
            yield self[i]

    def __mod__(self, value, /):
        return np_frontend.mod(self, value, out=self)

    def ptp(self, *, axis=None, out=None, keepdims=False):
        xmax = self.max(axis=axis, out=out, keepdims=keepdims)
        xmin = self.min(axis=axis, out=out, keepdims=keepdims)
        return np_frontend.subtract(xmax, xmin)

    def item(self, *args):
        if len(args) == 0:
            return self[0].ivy_array.to_scalar()
        elif len(args) == 1 and type(args[0]) == int:
            index = args[0]
            return self.ivy_array.flatten()[index].to_scalar()
        else:
            out = self
            for index in args:
                out = out[index]
            return out.ivy_array.to_scalar()

    def __rshift__(self, value, /):
        return ivy.bitwise_right_shift(self.ivy_array, value)

    def __lshift__(self, value, /):
        return ivy.bitwise_left_shift(self.ivy_array, value)
