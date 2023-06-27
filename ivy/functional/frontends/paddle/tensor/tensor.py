# local
import ivy
import ivy.functional.frontends.paddle as paddle_frontend
from ivy.func_wrapper import with_supported_dtypes, with_unsupported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import _to_ivy_array


class Tensor:
    def __init__(self, array, dtype=None, place="cpu", stop_gradient=True):
        self._ivy_array = (
            ivy.array(array, dtype=dtype, device=place)
            if not isinstance(array, ivy.Array)
            else array
        )
        self._dtype = dtype
        self._place = place
        self._stop_gradient = stop_gradient

    def __repr__(self):
        return (
            str(self._ivy_array.__repr__())
            .replace("ivy.array", "ivy.frontends.paddle.Tensor")
            .replace("dev", "place")
        )

    # Properties #
    # ---------- #

    @property
    def ivy_array(self):
        return self._ivy_array

    @property
    def place(self):
        return self.ivy_array.device

    @property
    def dtype(self):
        return self._ivy_array.dtype

    @property
    def shape(self):
        return self._ivy_array.shape

    @property
    def ndim(self):
        return self.dim()

    # Setters #
    # --------#

    @ivy_array.setter
    def ivy_array(self, array):
        self._ivy_array = (
            ivy.array(array) if not isinstance(array, ivy.Array) else array
        )

    # Special Methods #
    # -------------------#

    def __getitem__(self, item):
        ivy_args = ivy.nested_map([self, item], _to_ivy_array)
        ret = ivy.get_item(*ivy_args)
        return paddle_frontend.Tensor(ret)

    def __setitem__(self, item, value):
        item, value = ivy.nested_map([item, value], _to_ivy_array)
        self.ivy_array[item] = value

    def __iter__(self):
        if self.ndim == 0:
            raise TypeError("iteration over a 0-d tensor not supported")
        for i in range(self.shape[0]):
            yield self[i]

    # Instance Methods #
    # ---------------- #

    def reshape(self, *args, shape=None):
        if args and shape:
            raise TypeError("reshape() got multiple values for argument 'shape'")
        if shape is not None:
            return paddle_frontend.reshape(self._ivy_array, shape)
        if args:
            if isinstance(args[0], (tuple, list)):
                shape = args[0]
                return paddle_frontend.reshape(self._ivy_array, shape)
            else:
                return paddle_frontend.reshape(self._ivy_array, args)
        return paddle_frontend.reshape(self._ivy_array)

    def dim(self):
        return self.ivy_array.ndim

    @with_unsupported_dtypes({"2.5.0 and below": ("float16", "bfloat16")}, "paddle")
    def abs(self):
        return paddle_frontend.abs(self)

    @with_unsupported_dtypes({"2.5.0 and below": ("float16", "bfloat16")}, "paddle")
    def ceil(self):
        return paddle_frontend.ceil(self)

    @with_unsupported_dtypes({"2.5.0 and below": ("float16",)}, "paddle")
    def asinh(self, name=None):
        return ivy.asinh(self._ivy_array)

    @with_supported_dtypes({"2.5.0 and below": ("float32", "float64")}, "paddle")
    def asin(self, name=None):
        return ivy.asin(self._ivy_array)

    @with_supported_dtypes({"2.5.0 and below": ("float32", "float64")}, "paddle")
    def cosh(self, name=None):
        return ivy.cosh(self._ivy_array)

    @with_supported_dtypes({"2.5.0 and below": ("float32", "float64")}, "paddle")
    def log(self, name=None):
        return ivy.log(self._ivy_array)

    @with_supported_dtypes({"2.5.0 and below": ("float32", "float64")}, "paddle")
    def sin(self, name=None):
        return ivy.sin(self._ivy_array)

    @with_supported_dtypes({"2.5.0 and below": ("float32", "float64")}, "paddle")
    def sinh(self, name=None):
        return ivy.sinh(self._ivy_array)

    @with_unsupported_dtypes({"2.5.0 and below": ("float16", "bfloat16")}, "paddle")
    def argmax(self, axis=None, keepdim=False, dtype=None, name=None):
        return ivy.argmax(self._ivy_array, axis=axis, keepdims=keepdim, dtype=dtype)

    @with_supported_dtypes({"2.5.0 and below": ("float32", "float64")}, "paddle")
    def sqrt(self, name=None):
        return ivy.sqrt(self._ivy_array)

    @with_supported_dtypes({"2.5.0 and below": ("float32", "float64")}, "paddle")
    def cos(self, name=None):
        return ivy.cos(self._ivy_array)

    @with_unsupported_dtypes({"2.5.0 and below": ("float16", "bfloat16")}, "paddle")
    def exp(self, name=None):
        return ivy.exp(self._ivy_array)

    @with_supported_dtypes({"2.5.0 and below": ("float32", "float64")}, "paddle")
    def erf(self, name=None):
        return ivy.erf(self._ivy_array)

    @with_unsupported_dtypes({"2.5.0 and below": ("float16", "bfloat16")}, "paddle")
    def subtract(self, y, name=None):
        y_ivy = _to_ivy_array(y)
        return ivy.subtract(self._ivy_array, y_ivy)

    @with_unsupported_dtypes({"2.5.0 and below": ("float16", "bfloat16")}, "paddle")
    def log10(self, name=None):
        return ivy.log10(self._ivy_array)

    @with_unsupported_dtypes({"2.5.0 and below": ("float16", "bfloat16")}, "paddle")
    def argsort(self, axis=-1, descending=False, name=None):
        return ivy.argsort(self._ivy_array, axis=axis, descending=descending)

    @with_unsupported_dtypes({"2.5.0 and below": ("float16", "bfloat16")}, "paddle")
    def floor(self, name=None):
        return ivy.floor(self._ivy_array)

    @with_supported_dtypes({"2.5.0 and below": ("float32", "float64")}, "paddle")
    def tanh(self, name=None):
        return ivy.tanh(self._ivy_array)

    @with_supported_dtypes({"2.5.0 and below": ("float32", "float64")}, "paddle")
    def add_(self, name=None):
        return ivy.add(self._ivy_array)

    @with_supported_dtypes(
        {"2.5.0 and below": ("float16", "float32", "float64", "int32", "int64")},
        "paddle",
    )
    def isinf(self, name=None):
        return ivy.isinf(self._ivy_array)

    @with_supported_dtypes({"2.5.0 and below": ("float32", "float64")}, "paddle")
    def square(self, name=None):
        return ivy.square(self._ivy_array)

    @with_supported_dtypes({"2.5.0 and below": ("float32", "float64")}, "paddle")
    def cholesky(self, upper=False, name=None):
        return ivy.cholesky(self._ivy_array, upper=upper)

    @with_unsupported_dtypes({"2.5.0 and below": ("float16", "bfloat16")}, "paddle")
    def multiply(self, y, name=None):
        return paddle_frontend.multiply(self, y)

    @with_supported_dtypes(
        {"2.5.0 and below": ("float16", "float32", "float64", "int32", "int64")},
        "paddle",
    )
    def isfinite(self, name=None):
        return ivy.isfinite(self._ivy_array)

    def all(self, axis=None, keepdim=False, name=None):
        return ivy.all(self.ivy_array, axis=axis, keepdims=keepdim)

    @with_supported_dtypes({"2.5.0 and below": ("float16", "bfloat16")}, "paddle")
    def allclose(self, other, rtol=1e-05, atol=1e-08, equal_nan=False, name=None):
        return ivy.allclose(
            self._ivy_array, other, rtol=rtol, atol=atol, equal_nan=equal_nan
        )

    @with_unsupported_dtypes({"2.5.0 and below": ("float16", "bfloat16")}, "paddle")
    def sort(self, axis=-1, descending=False, name=None):
        return ivy.sort(self._ivy_array, axis=axis, descending=descending)

    @with_supported_dtypes(
        {
            "2.5.0 and below": (
                "bool",
                "int8",
                "int16",
                "int32",
                "int64",
                "float32",
                "float64",
            )
        },
        "paddle",
    )
    def logical_or(self, y, out=None, name=None):
        return paddle_frontend.logical_or(self, y, out=out)

    @with_supported_dtypes(
        {"2.5.0 and below": ("bool", "uint8", "int8", "int16", "int32", "int64")},
        "paddle",
    )
    def bitwise_xor(self, y, out=None, name=None):
        return paddle_frontend.bitwise_xor(self, y)

    @with_supported_dtypes({"2.5.0 and below": ("float16", "bfloat16")}, "paddle")
    def any(self, axis=None, keepdim=False, name=None):
        return ivy.any(self._ivy_array, axis=axis, keepdims=keepdim)

    @with_supported_dtypes({"2.5.0 and below": ("float16", "bfloat16")}, "paddle")
    def astype(self, dtype):
        return ivy.astype(self._ivy_array, dtype=dtype)

    @with_supported_dtypes(
        {
            "2.5.0 and below": (
                "bool",
                "int8",
                "int16",
                "int32",
                "int64",
                "float32",
                "float64",
            )
        },
        "paddle",
    )
    def logical_xor(self, y, out=None, name=None):
        return paddle_frontend.logical_xor(self, y, out=out)

    @with_unsupported_dtypes(
        {
            "2.5.0 and below": (
                "bool",
                "uint8",
                "int8",
                "int16",
                "complex64",
                "complex128",
            )
        },
        "paddle",
    )
    def greater_than(self, y, name=None):
        return paddle_frontend.greater_than(self, y)

    @with_supported_dtypes({"2.5.0 and below": ("float32", "float64")}, "paddle")
    def rsqrt(self, name=None):
        return ivy.reciprocal(ivy.sqrt(self._ivy_array))
