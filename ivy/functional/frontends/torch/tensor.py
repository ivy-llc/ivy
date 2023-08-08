# global
from typing import Iterable
import math

# local
import ivy
import ivy.functional.frontends.torch as torch_frontend
import ivy.functional.frontends.torch.nn.functional as torch_frontend_nn
from ivy.functional.frontends.numpy.creation_routines.from_existing_data import (
    array as np_frontend_array,
)
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.func_wrapper import with_supported_dtypes
from ivy.functional.frontends.torch.func_wrapper import (
    _to_ivy_array,
    numpy_to_torch_style_args,
)


class Tensor:
    def __init__(self, array, device=None, _init_overload=False, requires_grad=False):
        if _init_overload:
            self._ivy_array = (
                ivy.array(array) if not isinstance(array, ivy.Array) else array
            )

        else:
            self._ivy_array = ivy.array(
                array, dtype=torch_frontend.float32, device=device
            )
        self._grads = None
        self._requires_grad = requires_grad
        self.grad_fn = None
        if not _init_overload:
            self._is_leaf = True
        else:
            self._is_leaf = False
        self._requires_grad = requires_grad

    def __len__(self):
        return len(self._ivy_array)

    def __repr__(self):
        return str(self.ivy_array.__repr__()).replace(
            "ivy.array", "ivy.frontends.torch.Tensor"
        )

    # Properties #
    # ---------- #

    @property
    def ivy_array(self):
        return self._ivy_array

    @property
    def device(self):
        return self.ivy_array.device

    @property
    def dtype(self):
        return self.ivy_array.dtype

    @property
    def shape(self):
        return Size(self.ivy_array.shape)

    @property
    def real(self):
        return self.ivy_array.real()

    @property
    def imag(self):
        return self.ivy_array.imag()

    @property
    def ndim(self):
        return self.dim()

    @property
    def T(self):
        if self.ndim == 1:
            return self
        return torch_frontend.permute(self, list(range(self.ndim))[::-1])

    @property
    def data(self):
        return torch_frontend.tensor(
            ivy.stop_gradient(self.ivy_array, preserve_type=False)
        )

    @property
    def grad(self):
        return self._grads

    @property
    def requires_grad(self):
        return self._requires_grad

    @property
    def is_leaf(self):
        return self._is_leaf

    # Setters #
    # --------#

    @ivy_array.setter
    def ivy_array(self, array):
        self._ivy_array = (
            ivy.array(array) if not isinstance(array, ivy.Array) else array
        )

    @requires_grad.setter
    def requires_grad(self, requires_grad):
        self._requires_grad = requires_grad

    @is_leaf.setter
    def is_leaf(self, is_leaf):
        self._is_leaf = is_leaf

    # Instance Methods #
    # ---------------- #
    def reshape(self, *args, shape=None):
        if args and shape:
            raise TypeError("reshape() got multiple values for argument 'shape'")
        if shape is not None:
            return torch_frontend.reshape(self, shape)
        if args:
            if isinstance(args[0], (tuple, list, ivy.Shape)):
                shape = args[0]
                return torch_frontend.reshape(self, shape)
            else:
                return torch_frontend.reshape(self, args)
        return torch_frontend.reshape(self)

    @with_unsupported_dtypes({"2.0.1 and below": ("bfloat16",)}, "torch")
    def reshape_as(self, other):
        return torch_frontend.reshape(self, other.shape)

    @with_unsupported_dtypes({"2.0.1 and below": ("bfloat16",)}, "torch")
    def add(self, other, *, alpha=1):
        return torch_frontend.add(self, other, alpha=alpha)

    @with_unsupported_dtypes({"2.0.1 and below": ("bfloat16",)}, "torch")
    def sub(self, other, *, alpha=1):
        return torch_frontend.sub(self, other, alpha=alpha)

    def chunk(self, chunks, dim=0):
        return torch_frontend.chunk(self, chunks, dim=dim)

    @numpy_to_torch_style_args
    def any(self, dim=None, keepdim=False):
        return torch_frontend.any(self, dim=dim, keepdim=keepdim)

    @numpy_to_torch_style_args
    def all(self, dim=None, keepdim=False):
        return torch_frontend.all(self, dim=dim, keepdim=keepdim)

    @with_unsupported_dtypes({"2.0.1 and below": ("bfloat16",)}, "torch")
    def add_(self, other, *, alpha=1):
        self.ivy_array = self.add(other, alpha=alpha).ivy_array
        return self

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def addmm(self, mat1, mat2, *, beta=1, alpha=1):
        return torch_frontend.addmm(self, mat1, mat2, beta=beta, alpha=alpha)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def addmm_(self, mat1, mat2, *, beta=1, alpha=1):
        self.ivy_array = self.addmm(mat1, mat2, beta=beta, alpha=alpha).ivy_array
        return self

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def addmv(self, mat, vec, *, beta=1, alpha=1):
        return torch_frontend.addmv(self, mat, vec, beta=beta, alpha=alpha)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def addmv_(self, mat, vec, *, beta=1, alpha=1):
        self.ivy_array = torch_frontend.addmv(
            self, mat, vec, beta=beta, alpha=alpha
        ).ivy_array
        return self

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def addbmm(self, batch1, batch2, *, beta=1, alpha=1):
        return torch_frontend.addbmm(self, batch1, batch2, beta=beta, alpha=alpha)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def addbmm_(self, batch1, batch2, *, beta=1, alpha=1):
        self.ivy_array = self.addbmm(batch1, batch2, beta=beta, alpha=alpha).ivy_array
        return self

    @with_unsupported_dtypes({"2.0.1 and below": ("bfloat16",)}, "torch")
    def subtract_(self, other, *, alpha=1):
        self.ivy_array = self.sub(other, alpha=alpha).ivy_array
        return self

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def asin(self):
        return torch_frontend.asin(self)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def asin_(self):
        self.ivy_array = self.asin().ivy_array
        return self

    @numpy_to_torch_style_args
    @with_unsupported_dtypes({"2.0.1 and below": ("bfloat16",)}, "torch")
    def sum(self, dim=None, keepdim=False, *, dtype=None):
        return torch_frontend.sum(self, dim=dim, keepdim=keepdim, dtype=dtype)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def sin(self):
        return torch_frontend.sin(self)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def sin_(self):
        self.ivy_array = self.sin().ivy_array
        return self

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def sinh(self):
        return torch_frontend.sinh(self)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def sinh_(self):
        self.ivy_array = self.sinh().ivy_array
        return self

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def cos(self):
        return torch_frontend.cos(self)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def cos_(self):
        self.ivy_array = self.cos().ivy_array
        return self

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def cosh(self):
        return torch_frontend.cosh(self)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def cosh_(self):
        self.ivy_array = self.cosh().ivy_array
        return self

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def arcsinh(self):
        return torch_frontend.arcsinh(self)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def arcsin(self):
        return torch_frontend.arcsin(self)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def arcsin_(self):
        self.ivy_array = self.arcsin().ivy_array
        return self

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def atan(self):
        return torch_frontend.atan(self)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def atan_(self):
        self.ivy_array = self.atan().ivy_array
        return self

    @with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
    def atan2(self, other):
        return torch_frontend.atan2(self, other)

    def view(self, *args, size=None):
        """
        Reshape Tensor.

        possible arguments are either:
            - size
            - tuple of ints
            - list of ints
            - torch.Size object
            - ints
        Parameters
        ----------
        args:int arguments
        size: optional shape

        Returns reshaped tensor
        -------
        """
        if ivy.exists(size) and not args:
            shape_tup = size
        elif args and not ivy.exists(size):
            if (
                isinstance(args[0], (tuple, list, ivy.Shape))
                or type(args[0]).__name__ == "Size"
            ) and len(args) == 1:
                shape_tup = args[0]
            else:
                shape_tup = args
        else:
            raise ValueError(
                "View only accepts as argument ints, tuple or list of ints or "
                "the keyword argument size."
            )
        return torch_frontend.reshape(self, shape_tup)

    def float(self, memory_format=None):
        self.ivy_array = ivy.astype(self.ivy_array, ivy.float32, copy=False)
        return self

    def double(self):
        return self.to(torch_frontend.float64)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def asinh(self):
        return torch_frontend.asinh(self)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def asinh_(self):
        self.ivy_array = self.asinh().ivy_array
        return self

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def tan(self):
        return torch_frontend.tan(self)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def tan_(self):
        self.ivy_array = self.tan().ivy_array
        return self

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def tanh(self):
        return torch_frontend.tanh(self)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def tanh_(self):
        self.ivy_array = self.tanh().ivy_array
        return self

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def atanh(self):
        return torch_frontend.atanh(self)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def atanh_(self):
        self.ivy_array = self.atanh().ivy_array
        return self

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def arctanh(self):
        return torch_frontend.arctanh(self)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def arctanh_(self):
        self.ivy_array = self.arctanh().ivy_array
        return self

    @with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
    def log(self):
        return torch_frontend.log(self)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def arccosh(self):
        return torch_frontend.arccosh(self)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def log_(self):
        self.ivy_array = self.log().ivy_array
        return self

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def log2(self):
        return torch_frontend.log2(self)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
    def relu(self):
        return torch_frontend_nn.relu(self)

    @numpy_to_torch_style_args
    @with_unsupported_dtypes({"2.0.1 and below": ("complex",)}, "torch")
    def amax(self, dim=None, keepdim=False):
        return torch_frontend.amax(self, dim=dim, keepdim=keepdim)

    @numpy_to_torch_style_args
    @with_unsupported_dtypes({"2.0.1 and below": ("complex",)}, "torch")
    def amin(self, dim=None, keepdim=False):
        return torch_frontend.amin(self, dim=dim, keepdim=keepdim)

    @numpy_to_torch_style_args
    @with_unsupported_dtypes({"2.0.1 and below": ("complex", "float16")}, "torch")
    def aminmax(self, dim=None, keepdim=False):
        return torch_frontend.aminmax(self, dim=dim, keepdim=keepdim)

    def abs(self):
        return torch_frontend.abs(self)

    def abs_(self):
        self.ivy_array = self.abs().ivy_array
        return self

    @with_unsupported_dtypes({"2.0.1 and below": ("bfloat16",)}, "torch")
    def logical_and(self, other):
        return torch_frontend.logical_and(self, other)

    def logical_not(self, *, out=None):
        return torch_frontend.logical_not(self, out=out)

    def logical_not_(self):
        self.ivy_array = ivy.astype(self.logical_not().ivy_array, self.dtype)
        return self

    @with_unsupported_dtypes({"2.0.1 and below": ("bfloat16",)}, "torch")
    def logical_or(self, other):
        return torch_frontend.logical_or(self, other)

    def bitwise_not(self):
        return torch_frontend.bitwise_not(self)

    def bitwise_and(self, other):
        return torch_frontend.bitwise_and(self, other)

    @with_supported_dtypes({"2.0.1 and below": ("integer",)}, "torch")
    def bitwise_or(self, other):
        return torch_frontend.bitwise_or(self, other)

    def bitwise_left_shift(self, other):
        return torch_frontend.bitwise_left_shift(self, other)

    @with_supported_dtypes({"2.0.1 and below": ("integer",)}, "torch")
    def bitwise_or_(self, other):
        self.ivy_array = self.bitwise_or(other).ivy_array
        return self

    def contiguous(self, memory_format=None):
        return torch_frontend.tensor(self)

    def new_ones(
        self,
        *args,
        size=None,
        dtype=None,
        device=None,
        requires_grad=False,
        layout=None,
        pin_memory=False,
    ):
        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device
        if size is None:
            size = args[0] if isinstance(args[0], (tuple, list, ivy.Shape)) else args
        return torch_frontend.ones(
            size, dtype=dtype, device=device, requires_grad=requires_grad
        )

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def floor(self, *, out=None):
        return torch_frontend.floor(self)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
    def not_equal(self, other, *, out=None):
        return torch_frontend.not_equal(self, other, out=out)

    ne = not_equal

    def equal(self, other):
        return torch_frontend.equal(self, other)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16", "complex")}, "torch")
    def erf(self, *, out=None):
        return torch_frontend.erf(self, out=out)

    def new_zeros(
        self,
        size,
        *,
        dtype=None,
        device=None,
        requires_grad=False,
        layout=None,
        pin_memory=False,
    ):
        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device
        if isinstance(size[0], (tuple, list, ivy.Shape)):
            return torch_frontend.zeros(
                size=size[0], dtype=dtype, device=device, requires_grad=requires_grad
            )
        return torch_frontend.zeros(
            size=size, dtype=dtype, device=device, requires_grad=requires_grad
        )

    def to(self, *args, **kwargs):
        if len(args) > 0:
            if hasattr(args[0], "ivy_array") or ivy.is_array(args[0]):
                if self.dtype == ivy.dtype(args[0]) and self.device == ivy.dev(args[0]):
                    return self
                else:
                    cast_tensor = self.clone()
                    cast_tensor.ivy_array = ivy.asarray(
                        self.ivy_array,
                        dtype=ivy.dtype(args[0]),
                        device=ivy.dev(args[0]),
                    )
                    return cast_tensor
            if (
                isinstance(args[0], (ivy.Dtype, ivy.NativeDtype))
                or args[0] in ivy._all_ivy_dtypes_str
            ):
                if self.dtype == ivy.as_ivy_dtype(args[0]):
                    return self
                else:
                    cast_tensor = self.clone()
                    cast_tensor.ivy_array = ivy.asarray(self.ivy_array, dtype=args[0])
                    return cast_tensor
            if isinstance(args[0], (ivy.Device, ivy.NativeDevice, str)):
                if isinstance(args[0], str) and not isinstance(
                    args[0], (ivy.Device, ivy.NativeDevice)
                ):
                    ivy.utils.assertions.check_elem_in_list(
                        args[0],
                        [
                            "cpu",
                            "cuda",
                            "xpu",
                            "mkldnn",
                            "opengl",
                            "opencl",
                            "ideep",
                            "hip",
                            "ve",
                            "ort",
                            "mlc",
                            "xla",
                            "lazy",
                            "vulkan",
                            "meta",
                            "hpu",
                        ],
                    )
                if self.device == ivy.as_ivy_dev(args[0]):
                    return self
                else:
                    cast_tensor = self.clone()
                    cast_tensor.ivy_array = ivy.asarray(self.ivy_array, device=args[0])
                    return cast_tensor
        else:
            if (
                "dtype" in kwargs
                and "device" in kwargs
                and self.dtype == kwargs["dtype"]
                and self.device == kwargs["device"]
            ):
                return self
            else:
                cast_tensor = self.clone()
                cast_tensor.ivy_array = ivy.asarray(
                    self.ivy_array,
                    device=kwargs["device"] if "device" in kwargs else self.device,
                    dtype=kwargs["dtype"] if "dtype" in kwargs else self.dtype,
                )
                return cast_tensor

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def arctan(self):
        return torch_frontend.atan(self)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def arctan_(self):
        self.ivy_array = self.arctan().ivy_array
        return self

    @with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
    def arctan2(self, other):
        return torch_frontend.arctan2(self, other)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
    def arctan2_(self, other):
        self.ivy_array = self.arctan2(other).ivy_array
        return self

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def acos(self):
        return torch_frontend.acos(self)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def acos_(self):
        self.ivy_array = self.acos().ivy_array
        return self

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def arccosh_(self):
        self.ivy_array = self.arccosh().ivy_array
        return self

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def arccos(self):
        return torch_frontend.arccos(self)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def arccos_(self):
        self.ivy_array = self.arccos().ivy_array
        return self

    def new_tensor(
        self,
        data,
        *,
        dtype=None,
        device=None,
        requires_grad=False,
        layout=None,
        pin_memory=False,
    ):
        dtype = ivy.dtype(self.ivy_array) if dtype is None else dtype
        device = ivy.dev(self.ivy_array) if device is None else device
        _data = ivy.asarray(data, copy=True, dtype=dtype, device=device)
        return torch_frontend.tensor(_data)

    @with_unsupported_dtypes({"2.0.1 and below": ("bfloat16",)}, "torch")
    def view_as(self, other):
        return self.view(size=other.shape)

    def expand(self, *args, size=None):
        if args and size:
            raise TypeError("expand() got multiple values for argument 'size'")
        if args:
            if isinstance(args[0], (tuple, list, ivy.Shape)):
                size = args[0]
            else:
                size = args

        return torch_frontend.tensor(ivy.expand(self.ivy_array, tuple(size)))

    def expand_as(self, other):
        return self.expand(
            ivy.shape(other.ivy_array if isinstance(other, Tensor) else other)
        )

    def detach(self):
        return torch_frontend.tensor(
            ivy.stop_gradient(self.ivy_array, preserve_type=False)
        )

    def detach_(self):
        self.ivy_array = self.detach().ivy_array
        return self

    @numpy_to_torch_style_args
    def unsqueeze(self, dim):
        return torch_frontend.unsqueeze(self, dim)

    @numpy_to_torch_style_args
    def unsqueeze_(self, dim):
        self.ivy_array = self.unsqueeze(dim).ivy_array
        return self

    def ravel(self):
        return torch_frontend.ravel(self)

    def split(self, split_size, dim=0):
        return torch_frontend.split(self, split_size, dim)

    def tensor_split(self, indices_or_sections, dim=0):
        return torch_frontend.tensor_split(self, indices_or_sections, dim)

    def vsplit(self, indices_or_sections, /):
        return torch_frontend.vsplit(self, indices_or_sections)

    def hsplit(self, indices_or_sections, /):
        return torch_frontend.hsplit(self, indices_or_sections)

    def dsplit(
        self,
        indices_or_sections,
        /,
    ):
        return torch_frontend.dsplit(self, indices_or_sections)

    def dim(self):
        return self.ivy_array.ndim

    @with_supported_dtypes(
        {"2.5.0 and below": ("float32", "float64", "int32", "int64")}, "paddle"
    )
    def heaviside(self, values, *, out=None):
        return torch_frontend.heaviside(self, values, out=out)

    def new_full(
        self,
        size,
        fill_value,
        *,
        dtype=None,
        device=None,
        requires_grad=False,
        layout=None,
        pin_memory=False,
    ):
        dtype = ivy.dtype(self.ivy_array) if dtype is None else dtype
        if ivy.is_float_dtype(dtype):
            fill_value = float(fill_value)
        elif ivy.is_int_dtype(dtype):
            fill_value = int(fill_value)
        elif ivy.is_bool_dtype(dtype):
            fill_value = bool(fill_value)
        device = ivy.dev(self.ivy_array) if device is None else device
        _data = ivy.full(size, fill_value, dtype=dtype, device=device)
        return torch_frontend.tensor(_data)

    def new_empty(
        self,
        size,
        *,
        dtype=None,
        device=None,
        requires_grad=False,
        layout=None,
        pin_memory=False,
    ):
        dtype = ivy.dtype(self.ivy_array) if dtype is None else dtype
        device = ivy.dev(self.ivy_array) if device is None else device
        _data = ivy.empty(size, dtype=dtype, device=device)
        return torch_frontend.tensor(_data)

    def unfold(self, dimension, size, step):
        slices = []
        for i in range(0, self.shape[dimension] - size + 1, step):
            slices.append(self.ivy_array[i : i + size])
        return torch_frontend.stack(slices)

    def long(self, memory_format=None):
        self.ivy_array = ivy.astype(self.ivy_array, ivy.int64, copy=False)
        return self

    @numpy_to_torch_style_args
    def max(self, dim=None, keepdim=False):
        return torch_frontend.max(self, dim=dim, keepdim=keepdim)

    @property
    def is_quantized(self):
        return "q" in ivy.dtype(self.ivy_array)

    @property
    def is_cuda(self):
        return "gpu" in ivy.dev(self.ivy_array)

    @property
    def is_meta(self):
        return "meta" in ivy.dev(self.ivy_array)

    @with_unsupported_dtypes({"2.0.1 and below": ("bfloat16",)}, "torch")
    def pow(self, exponent):
        return torch_frontend.pow(self, exponent)

    @with_unsupported_dtypes({"2.0.1 and below": ("bfloat16",)}, "torch")
    def pow_(self, exponent):
        self.ivy_array = self.pow(exponent).ivy_array
        return self

    def size(self, dim=None):
        shape = self.shape
        if dim is None:
            return shape
        else:
            try:
                return shape[dim]
            except IndexError:
                raise IndexError(
                    "Dimension out of range (expected to be in range of [{}, {}], "
                    "but got {}".format(len(shape), len(shape) - 1, dim)
                )

    def matmul(self, other):
        return torch_frontend.matmul(self, other)

    def argwhere(self):
        return torch_frontend.argwhere(self)

    @numpy_to_torch_style_args
    @with_unsupported_dtypes({"2.0.1 and below": ("complex",)}, "torch")
    def argmax(self, dim=None, keepdim=False):
        return torch_frontend.argmax(self, dim=dim, keepdim=keepdim)

    @numpy_to_torch_style_args
    @with_unsupported_dtypes({"2.0.1 and below": ("complex",)}, "torch")
    def argmin(self, dim=None, keepdim=False):
        return torch_frontend.argmin(self, dim=dim, keepdim=keepdim)

    @with_unsupported_dtypes({"2.0.1 and below": ("complex",)}, "torch")
    def argsort(self, dim=-1, descending=False):
        return torch_frontend.argsort(self, dim=dim, descending=descending)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def ceil(self):
        return torch_frontend.ceil(self)

    @numpy_to_torch_style_args
    def min(self, dim=None, keepdim=False):
        return torch_frontend.min(self, dim=dim, keepdim=keepdim)

    def permute(self, *args, dims=None):
        if args and dims:
            raise TypeError("permute() got multiple values for argument 'dims'")
        if dims is not None:
            return torch_frontend.permute(self, dims)
        if args:
            if isinstance(args[0], (tuple, list, ivy.Shape)):
                dims = args[0]
                return torch_frontend.permute(self, dims)
            else:
                return torch_frontend.permute(self, args)
        return torch_frontend.permute(self)

    @numpy_to_torch_style_args
    @with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
    def mean(self, dim=None, keepdim=False):
        return torch_frontend.mean(self, dim=dim, keepdim=keepdim)

    @numpy_to_torch_style_args
    def nanmean(self, dim=None, keepdim=False):
        return torch_frontend.nanmean(self, dim=dim, keepdim=keepdim)

    @numpy_to_torch_style_args
    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def median(self, dim=None, keepdim=False):
        return torch_frontend.median(self, dim=dim, keepdim=keepdim)

    def transpose(self, dim0, dim1):
        return torch_frontend.transpose(self, dim0=dim0, dim1=dim1)

    def transpose_(self, dim0, dim1):
        self.ivy_array = self.transpose(dim0, dim1).ivy_array
        return self

    def t(self):
        return torch_frontend.t(self)

    def flatten(self, start_dim=0, end_dim=-1):
        return torch_frontend.flatten(self, start_dim, end_dim)

    @numpy_to_torch_style_args
    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def cumsum(self, dim, *, dtype=None):
        return torch_frontend.cumsum(self, dim, dtype=dtype)

    @numpy_to_torch_style_args
    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def cumsum_(self, dim, *, dtype=None):
        self.ivy_array = self.cumsum(dim, dtype).ivy_array
        return self

    @with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
    def inverse(self):
        return torch_frontend.inverse(self)

    @with_unsupported_dtypes({"2.0.1 and below": ("bool",)}, "torch")
    def neg(self):
        return torch_frontend.negative(self)

    __neg__ = neg

    def int(self, memory_format=None):
        self.ivy_array = ivy.astype(self.ivy_array, ivy.int32, copy=False)
        return self

    def half(self, memory_format=None):
        self.ivy_array = ivy.astype(self.ivy_array, ivy.float16, copy=False)
        return self

    def bool(self, memory_format=None):
        self.ivy_array = ivy.astype(self.ivy_array, ivy.bool, copy=False)
        return self

    def type(self, dtype=None, non_blocking=False, **kwargs):
        if ivy.exists(dtype):
            self.ivy_array = ivy.astype(self.ivy_array, dtype)
            return self
        else:
            return str(self.dtype)

    @with_unsupported_dtypes({"2.0.1 and below": ("bfloat16",)}, "torch")
    def type_as(self, other):
        if self.dtype != other.dtype:
            self.ivy_array = ivy.astype(self.ivy_array, other.dtype)
        return self

    def byte(self, memory_format=None):
        self.ivy_array = ivy.astype(self.ivy_array, ivy.uint8, copy=False)
        return self

    @with_unsupported_dtypes({"2.0.1 and below": ("bfloat16",)}, "torch")
    def ne(self, other):
        return torch_frontend.ne(self, other)

    @numpy_to_torch_style_args
    def squeeze(self, dim=None):
        return torch_frontend.squeeze(self, dim)

    @numpy_to_torch_style_args
    @with_unsupported_dtypes({"2.0.1 and below": ("bfloat16", "uint16")}, "torch")
    def squeeze_(self, dim=None):
        self.ivy_array = self.squeeze(dim).ivy_array
        return self

    def flip(self, dims):
        return torch_frontend.flip(self, dims)

    def fliplr(self):
        return torch_frontend.fliplr(self)

    def sort(self, dim=-1, descending=False):
        return torch_frontend.sort(self, dim=dim, descending=descending)

    def tril(self, diagonal=0):
        return torch_frontend.tril(self, diagonal=diagonal)

    def tril_(self, diagonal=0):
        self.ivy_array = self.tril(diagonal=diagonal).ivy_array
        return self

    def index_select(self, dim, index):
        return torch_frontend.index_select(self, dim, index)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16", "complex")}, "torch")
    def clamp(self, min=None, max=None):
        return torch_frontend.clamp(self, min=min, max=max)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16", "complex")}, "torch")
    def clamp_(self, min=None, max=None):
        self.ivy_array = self.clamp(min=min, max=max).ivy_array
        return self

    @with_unsupported_dtypes(
        {"2.0.1 and below": ("bool", "bfloat16", "float16", "complex")}, "torch"
    )
    def clamp_min(self, min=None):
        return torch_frontend.clamp(self, min=min)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
    def sqrt(self):
        return torch_frontend.sqrt(self)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def rsqrt(self):
        return torch_frontend.rsqrt(self)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
    def sqrt_(self):
        self.ivy_array = self.sqrt().ivy_array
        return self

    def where(self, condition, other):
        return torch_frontend.tensor(torch_frontend.where(condition, self, other))

    def clone(self, memory_format=None):
        return torch_frontend.tensor(ivy.array(self.ivy_array, copy=True))

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def acosh(self):
        return torch_frontend.acosh(self)

    def masked_fill(self, mask, value):
        return torch_frontend.tensor(
            torch_frontend.where(mask, value, self), dtype=self.dtype
        )

    def masked_fill_(self, mask, value):
        self.ivy_array = self.masked_fill(mask, value).ivy_array
        return self

    @with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
    def index_add_(self, dim, index, source, *, alpha=1):
        self.ivy_array = torch_frontend.index_add(
            self, dim, index, source, alpha=alpha
        ).ivy_array
        return self

    @with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
    def index_add(self, dim, index, source, *, alpha=1):
        return torch_frontend.index_add(
            self._ivy_array, dim, index, source, alpha=alpha
        )

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def acosh_(self):
        self.ivy_array = self.acosh().ivy_array
        return self

    @with_unsupported_dtypes({"2.0.1 and below": ("bfloat16",)}, "torch")
    def numpy(self):
        return np_frontend_array(self.ivy_array)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def sigmoid(self):
        return torch_frontend.sigmoid(self)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def sigmoid_(self):
        self.ivy_array = self.sigmoid().ivy_array
        return self

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def softmax(self, dim=None, dtype=None):
        return torch_frontend.nn.functional.softmax(self, dim=dim, dtype=dtype)

    def repeat_interleave(self, repeats, dim=None, *, output_size=None):
        return torch_frontend.repeat_interleave(self, repeats, dim)

    def repeat(self, *args, repeats=None):
        if args and repeats:
            raise ivy.utils.exceptions.IvyException(
                "repeat() got multiple values for argument 'repeats'"
            )
        if args:
            if isinstance(args[0], (tuple, list, ivy.Shape)):
                repeats = args[0]
            else:
                repeats = args
        elif not isinstance(repeats, (tuple, list)):
            raise ivy.utils.exceptions.IvyException(
                "repeat(): argument 'repeats' must be tuple of ints"
            )

        return torch_frontend.tile(self, repeats)

    @numpy_to_torch_style_args
    def unbind(self, dim=0):
        return torch_frontend.unbind(self, dim=dim)

    def remainder(self, other, *, out=None):
        return torch_frontend.remainder(self, other, out=out)

    def bitwise_not_(self):
        self.ivy_array = self.bitwise_not().ivy_array
        return self

    def bitwise_and_(self, other):
        self.ivy_array = self.bitwise_and(other).ivy_array
        return self

    @with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
    def atan2_(self, other):
        self.ivy_array = self.atan2(other).ivy_array
        return self

    def fmin(self, other):
        return torch_frontend.fmin(self, other)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16", "complex")}, "torch")
    def trunc(self):
        return torch_frontend.trunc(self)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16", "complex")}, "torch")
    def trunc_(self):
        self.ivy_array = self.trunc().ivy_array
        return self

    @with_unsupported_dtypes({"2.0.1 and below": ("float16", "complex")}, "torch")
    def fix(self):
        return torch_frontend.fix(self)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16", "complex")}, "torch")
    def fix_(self):
        self.ivy_array = self.fix().ivy_array
        return self

    def isinf(self):
        return torch_frontend.isinf(self._ivy_array)

    def is_complex(self):
        return torch_frontend.is_complex(self._ivy_array)

    def addr(self, vec1, vec2, *, beta=1, alpha=1, out=None):
        return torch_frontend.addr(self, vec1, vec2, beta=beta, alpha=alpha, out=out)

    def addr_(self, vec1, vec2, *, beta=1, alpha=1):
        self.ivy_array = self.addr(vec1, vec2, beta=beta, alpha=alpha).ivy_array
        return self

    @with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
    def dot(self, tensor):
        return torch_frontend.dot(self, tensor)

    @with_supported_dtypes({"2.0.1 and below": ("float32", "float64")}, "torch")
    def bernoulli(self, *, generator=None, out=None):
        return torch_frontend.bernoulli(self._ivy_array, generator=generator, out=out)

    # Special Methods #
    # -------------------#

    def __bool__(self):
        if len(self.shape) == sum(self.shape):
            return torch_frontend.tensor(self.ivy_array.to_scalar().__bool__())
        raise ValueError(
            "The truth value of an array with more than one element is ambiguous. "
            "Use a.any() or a.all()"
        )

    @with_unsupported_dtypes({"2.0.1 and below": ("bfloat16",)}, "torch")
    def __add__(self, other):
        return torch_frontend.add(self, other)

    @with_unsupported_dtypes({"2.0.1 and below": ("bfloat16",)}, "torch")
    def __mod__(self, other):
        return torch_frontend.remainder(self, other)

    @with_unsupported_dtypes({"2.0.1 and below": ("bfloat16",)}, "torch")
    def __pow__(self, exponent):
        return self.pow(exponent)

    @with_unsupported_dtypes({"2.0.1 and below": ("bfloat16",)}, "torch")
    def __rpow__(self, other):
        return torch_frontend.pow(other, self)

    def __long__(self, memory_format=None):
        return self.long()

    def __getitem__(self, query, /):
        ivy_args = ivy.nested_map([self, query], _to_ivy_array)
        ret = ivy.get_item(*ivy_args)
        return torch_frontend.Tensor(ret, _init_overload=True)

    def __setitem__(self, key, value, /):
        key, value = ivy.nested_map([key, value], _to_ivy_array)
        self.ivy_array[key] = value

    def __iter__(self):
        if self.ndim == 0:
            raise TypeError("iteration over a 0-d tensor not supported")
        for i in range(self.shape[0]):
            yield self[i]

    @with_unsupported_dtypes({"2.0.1 and below": ("bfloat16",)}, "torch")
    def __radd__(self, other):
        return torch_frontend.add(other, self)

    @with_unsupported_dtypes({"2.0.1 and below": ("bfloat16",)}, "torch")
    def __mul__(self, other):
        return torch_frontend.mul(self, other)

    @with_unsupported_dtypes({"2.0.1 and below": "bfloat16"}, "torch")
    def __matmul__(self, other):
        return torch_frontend.matmul(self, other)

    @with_unsupported_dtypes({"2.0.1 and below": ("bfloat16",)}, "torch")
    def __rmul__(self, other):
        return torch_frontend.mul(other, self)

    @with_unsupported_dtypes({"2.0.1 and below": ("bfloat16",)}, "torch")
    def __sub__(self, other):
        return torch_frontend.subtract(self, other)

    def __truediv__(self, other):
        return torch_frontend.div(self, other)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16", "complex")}, "torch")
    def __floordiv__(self, other):
        return torch_frontend.floor_divide(self, other)

    def __iadd__(self, other):
        ret = torch_frontend.add(self, other)
        self.ivy_array = ivy.inplace_update(
            self.ivy_array, ivy.astype(ret.ivy_array, self.dtype)
        )
        return self

    def __imod__(self, other):
        ret = torch_frontend.remainder(self, other)
        self.ivy_array = ivy.inplace_update(
            self.ivy_array, ivy.astype(ret.ivy_array, self.dtype)
        )
        return self

    def __imul__(self, other):
        ret = torch_frontend.mul(self, other)
        self.ivy_array = ivy.inplace_update(
            self.ivy_array, ivy.astype(ret.ivy_array, self.dtype)
        )
        return self

    def __isub__(self, other):
        ret = torch_frontend.subtract(self, other)
        self.ivy_array = ivy.inplace_update(
            self.ivy_array, ivy.astype(ret.ivy_array, self.dtype)
        )
        return self

    def __itruediv__(self, other):
        ret = torch_frontend.div(self, other)
        self.ivy_array = ivy.inplace_update(
            self.ivy_array, ivy.astype(ret.ivy_array, self.dtype)
        )
        return self

    def __int__(self):
        item = self.item()
        if isinstance(item, complex):
            if item.imag != 0:
                raise TypeError("can't convert complex to int without overflow")
            item = item.real
        return int(item)

    def __float__(self):
        item = self.item()
        if isinstance(item, complex):
            if item.imag != 0:
                raise TypeError("can't convert complex to float without overflow")
            item = item.real
        return float(item)

    @with_unsupported_dtypes({"2.0.1 and below": ("bfloat16",)}, "torch")
    def __eq__(self, other):
        return torch_frontend.eq(self, other)

    @with_unsupported_dtypes({"2.0.1 and below": ("bfloat16",)}, "torch")
    def __gt__(self, other):
        return torch_frontend.greater(self, other)

    @with_unsupported_dtypes({"2.0.1 and below": ("bfloat16",)}, "torch")
    def __ge__(self, other):
        return torch_frontend.greater_equal(self, other)

    @with_unsupported_dtypes({"2.0.1 and below": ("bfloat16",)}, "torch")
    def __ne__(self, other):
        return self.ne(other)

    @with_unsupported_dtypes({"2.0.1 and below": ("bfloat16",)}, "torch")
    def __rsub__(self, other):
        return torch_frontend.subtract(other, self)

    @with_unsupported_dtypes({"2.0.1 and below": ("bfloat16",)}, "torch")
    def __lt__(self, other):
        return torch_frontend.less(self, other)

    @with_unsupported_dtypes({"2.0.1 and below": ("bfloat16",)}, "torch")
    def __le__(self, other):
        return torch_frontend.less_equal(self, other)

    @with_unsupported_dtypes({"2.0.1 and below": ("bfloat16",)}, "torch")
    def __or__(self, other):
        return torch_frontend.bitwise_or(self, other)

    def __invert__(self):
        return torch_frontend.bitwise_not(self)

    def __and__(self, other):
        return torch_frontend.bitwise_and(self, other)

    # Method aliases
    absolute, absolute_ = abs, abs_
    clip, clip_ = clamp, clamp_
    ndimension = dim
    subtract = sub
    sub_ = subtract_
    eq = equal

    def bitwise_xor(self, other):
        return torch_frontend.bitwise_xor(self, other)

    def item(self):
        if all(dim == 1 for dim in self.shape):
            return self.ivy_array.to_scalar()
        else:
            raise ValueError(
                "only one element tensors can be converted to Python scalars"
            )

    @numpy_to_torch_style_args
    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def cumprod(self, dim, dtype):
        return torch_frontend.cumprod(self, dim, dtype=dtype)

    @numpy_to_torch_style_args
    def count_nonzero(self, dim):
        return torch_frontend.count_nonzero(self, dim=dim)

    @with_unsupported_dtypes({"2.0.1 and below": ("bfloat16", "float16")}, "torch")
    def exp(self):
        return torch_frontend.exp(self)

    @with_unsupported_dtypes(
        {"2.0.1 and below": ("bfloat16", "float16", "complex")}, "torch"
    )
    def expm1(self):
        return torch_frontend.expm1(self)

    # fmt: off
    @with_unsupported_dtypes({"2.0.1 and below": ("int8", "int16", "int32", "int64", "uint8", "bool", "float16",)},"torch",)  # noqa
    def exp_(self):
        self.ivy_array = self.exp().ivy_array
        return self
    # fmt: on

    def mul(self, other):
        return torch_frontend.mul(self, other)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def ceil_(self):
        self.ivy_array = torch_frontend.ceil(self).ivy_array
        return self

    @with_unsupported_dtypes({"2.0.1 and below": ("bfloat16",)}, "torch")
    def mul_(self, other):
        self.ivy_array = self.mul(other).ivy_array
        # the return dtype is the same as the input dtype
        self.ivy_array = self.to(self.dtype).ivy_array
        return self

    @with_unsupported_dtypes({"2.0.1 and below": ("bfloat16", "float16")}, "torch")
    def round(self, *, decimals=0):
        return torch_frontend.round(self, decimals=decimals)

    @numpy_to_torch_style_args
    @with_unsupported_dtypes({"2.0.1 and below": ("float16", "complex")}, "torch")
    def cross(self, other, dim=-1):
        return torch_frontend.cross(self, other, dim=dim)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
    def det(self):
        return torch_frontend.det(self)

    def reciprocal(self):
        return torch_frontend.reciprocal(self)

    def fill_(self, value):
        self.ivy_array = torch_frontend.full_like(
            self, value, dtype=self.dtype, device=self.device
        ).ivy_array
        return self

    def nonzero(self, as_tuple=False):
        return torch_frontend.nonzero(self, as_tuple=as_tuple)

    def mm(self, mat2):
        return torch_frontend.mm(self, mat2)

    @with_unsupported_dtypes({"2.0.1 and below": ("bfloat16", "float16")}, "torch")
    def square(self):
        return torch_frontend.square(self._ivy_array)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
    def log10(self):
        return torch_frontend.log10(self._ivy_array)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def log10_(self):
        self.ivy_array = self.log10().ivy_array
        return self

    @with_unsupported_dtypes({"2.0.1 and below": ("bfloat16", "uint16")}, "torch")
    def zero_(self):
        self.ivy_array = torch_frontend.zeros_like(self).ivy_array
        return self

    def short(self, memory_format=None):
        self.ivy_array = ivy.astype(self.ivy_array, ivy.int16, copy=False)
        return self

    @numpy_to_torch_style_args
    @with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
    def prod(self, dim=None, keepdim=False, *, dtype=None):
        return torch_frontend.prod(self, dim=dim, keepdim=keepdim, dtype=dtype)

    def div(self, other, *, rounding_mode=None):
        return torch_frontend.div(self, other, rounding_mode=rounding_mode)

    def div_(self, other, *, rounding_mode=None):
        self.ivy_array = self.div(other, rounding_mode=rounding_mode).ivy_array
        return self

    def normal_(self, mean=0, std=1, *, generator=None):
        self.ivy_array = ivy.random_normal(
            mean=mean,
            std=std,
            shape=self.ivy_array.shape,
            dtype=self.dtype,
            device=self.device,
        )
        return self

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def addcdiv(self, tensor1, tensor2, *, value=1):
        return torch_frontend.addcdiv(self, tensor1, tensor2, value=value)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def addcmul(self, tensor1, tensor2, *, value=1):
        return torch_frontend.addcmul(self, tensor1, tensor2, value=value)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def addcmul_(self, tensor1, tensor2, *, value=1):
        self.ivy_array = self.addcmul(tensor1, tensor2, value=value).ivy_array
        return self

    sign_decorator_dtypes = ("float16", "complex", "bool")

    @with_unsupported_dtypes({"2.0.1 and below": sign_decorator_dtypes}, "torch")
    def sign(self):
        return torch_frontend.sign(self._ivy_array)

    @with_unsupported_dtypes({"2.0.1 and below": sign_decorator_dtypes}, "torch")
    def sign_(self):
        self.ivy_array = self.sign().ivy_array
        return self

    @numpy_to_torch_style_args
    def std(self, dim=None, unbiased=True, keepdim=False, *, out=None):
        return torch_frontend.std(
            self, dim=dim, unbiased=unbiased, keepdim=keepdim, out=out
        )

    @with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
    def fmod(self, other, *, out=None):
        return torch_frontend.fmod(self, other, out=out)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
    def fmod_(self, other):
        self.ivy_array = self.fmod(other).ivy_array
        return self

    def norm(self, p="fro", dim=None, keepdim=False, dtype=None):
        return torch_frontend.norm(self, p=p, dim=dim, keepdim=keepdim, dtype=dtype)

    def tolist(self):
        return self._ivy_array.to_list()

    @with_unsupported_dtypes({"2.0.1 and below": ("bfloat16",)}, "torch")
    def multiply(self, other, *, out=None):
        return torch_frontend.multiply(self, other, out=out)

    @numpy_to_torch_style_args
    @with_unsupported_dtypes({"2.0.1 and below": ("float16", "complex")}, "torch")
    def topk(self, k, dim=None, largest=True, sorted=True):
        return torch_frontend.topk(self, k, dim=dim, largest=largest, sorted=sorted)

    rshift_dtypes = ("float16", "bfloat16", "float32", "float64", "bool", "complex")

    @with_unsupported_dtypes({"2.0.1 and below": rshift_dtypes}, "torch")
    def bitwise_right_shift(self, other, *, out=None):
        return torch_frontend.bitwise_right_shift(self._ivy_array, other)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
    def logdet(self):
        chol = torch_frontend.cholesky(self)
        return 2 * torch_frontend.sum(
            torch_frontend.log(torch_frontend.real(torch_frontend.diagonal(chol)))
        )

    @with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
    def copysign(self, other, *, out=None):
        return torch_frontend.copysign(self, other, out=out)

    @with_unsupported_dtypes(
        {"2.0.1 and below": ("complex", "bfloat16", "bool")}, "torch"
    )
    def greater(self, other, *, out=None):
        return torch_frontend.greater(self, other, out=out)

    gt = greater

    @with_unsupported_dtypes({"2.0.1 and below": ("bfloat16", "bool")}, "torch")
    def greater_(self, other):
        self.ivy_array = ivy.astype(self.greater(other).ivy_array, self.dtype)
        return self

    gt_ = greater_

    @with_unsupported_dtypes(
        {"2.0.1 and below": ("complex", "bfloat16", "bool")}, "torch"
    )
    def greater_equal(self, other, *, out=None):
        return torch_frontend.greater_equal(self, other, out=out)

    ge = greater_equal

    @with_unsupported_dtypes({"2.0.1 and below": ("bfloat16", "bool")}, "torch")
    def greater_equal_(self, other):
        self.ivy_array = ivy.astype(self.greater_equal(other).ivy_array, self.dtype)
        return self

    ge_ = greater_equal_

    @with_unsupported_dtypes(
        {"2.0.1 and below": ("complex", "bfloat16", "bool")}, "torch"
    )
    def less(self, other, *, out=None):
        return torch_frontend.less(self, other, out=out)

    lt = less

    @with_unsupported_dtypes({"2.0.1 and below": ("bfloat16", "bool")}, "torch")
    def less_(self, other):
        self.ivy_array = ivy.astype(self.less(other).ivy_array, self.dtype)
        return self

    lt_ = less_

    @with_unsupported_dtypes(
        {"2.0.1 and below": ("complex", "bfloat16", "bool")}, "torch"
    )
    def less_equal(self, other, *, out=None):
        return torch_frontend.less_equal(self, other, out=out)

    le = less_equal

    @with_unsupported_dtypes({"2.0.1 and below": ("bfloat16", "bool")}, "torch")
    def less_equal_(self, other):
        self.ivy_array = ivy.astype(self.less_equal(other).ivy_array, self.dtype)
        return self

    le_ = less_equal_

    @with_unsupported_dtypes({"2.0.1 and below": ("bfloat16",)}, "torch")
    def eq_(self, other):
        self.ivy_array = ivy.astype(
            torch_frontend.eq(self, other).ivy_array, self.dtype
        )
        return self

    @numpy_to_torch_style_args
    def var(self, dim=None, *, correction=1, keepdim=False):
        return torch_frontend.var(self, dim=dim, unbiased=correction, keepdim=keepdim)

    def narrow(self, dim, start, length):
        return torch_frontend.narrow(self, dim=dim, start=start, length=length)

    def as_strided(self, size, stride, storage_offset=None):
        return torch_frontend.as_strided(
            self, size=size, stride=stride, storage_offset=storage_offset
        )

    def stride(self, dim=None):
        strides = [
            stride // math.ceil(ivy.dtype_bits(self.dtype) / 8)
            for stride in self.ivy_array.strides
        ]
        if dim is not None:
            return strides[dim]
        return strides

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def log1p(self):
        return torch_frontend.log1p(self)

    def baddbmm(self, batch1, batch2, *, beta=1, alpha=1):
        return torch_frontend.baddbmm(
            self, batch1=batch1, batch2=batch2, beta=beta, alpha=alpha
        )

    def bmm(self, mat2):
        return torch_frontend.bmm(self, mat2=mat2)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def floor_(self):
        self.ivy_array = self.floor().ivy_array
        return self

    def diag(self, diagonal=0):
        return torch_frontend.diag(self, diagonal=diagonal)

    @with_unsupported_dtypes({"2.0.1 and below": ("bfloat16",)}, "torch")
    def diagonal(self, offset=0, dim1=0, dim2=1):
        return torch_frontend.diagonal(self, offset=offset, dim1=dim1, dim2=dim2)

    def gather(self, dim, index):
        return torch_frontend.gather(self, dim=dim, index=index)

    def take_along_dim(self, indices, dim):
        return torch_frontend.take_along_dim(self, indices=indices, dim=dim)

    def movedim(self, source, destination):
        return torch_frontend.movedim(self, source=source, destination=destination)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16",)}, "torch")
    def addcdiv_(self, tensor1, tensor2, *, value=1):
        self.ivy_array = self.addcdiv(
            tensor1=tensor1, tensor2=tensor2, value=value
        ).ivy_array
        return self

    @with_unsupported_dtypes({"2.0.1 and below": ("bfloat16", "float16")}, "torch")
    def cholesky(self, upper=False):
        return torch_frontend.cholesky(self, upper=upper)

    def tile(self, *reps):
        if (
            isinstance(reps, Iterable)
            and len(reps) == 1
            and isinstance(reps[0], Iterable)
        ):
            reps = reps[0]
        return torch_frontend.tile(self, reps)

    def apply_(self, callable, /):
        if self.device != "cpu":
            raise Exception("apply_ is only supported on cpu tensors")
        self.ivy_array = callable(self.ivy_array)
        return self

    def requires_grad_(self, requires_grad=True):
        self._requires_grad = requires_grad
        return self

    def backward(self, gradient=None, retain_graph=None, create_graph=False):
        if gradient is None and int(torch_frontend.numel(self)) > 1:
            raise RuntimeError("grad can be implicitly created only for scalar outputs")
        if self.grad_fn is None and self._grads is None:
            assert self.shape == gradient.shape, "Mismatch in shape"
            self._grads = gradient
            return
        _grad_list = self.grad_fn(
            gradient if gradient is not None else torch_frontend.tensor(1.0)
        )
        for idx, next_function in enumerate(self.grad_fn.next_functions):
            if next_function.__self__.grad_fn is not None:
                next_function.__self__.backward(_grad_list[idx])
            else:
                next_function(_grad_list[idx])

    @with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
    def logaddexp(self, other):
        return torch_frontend.logaddexp(self, other)

    def angle(self):
        return torch_frontend.angle(self)

    @with_supported_dtypes(
        {
            "2.5.0 and below": (
                "int64",
                "float64",
                "complex128",
                "float32",
                "complex64",
                "int32",
            )
        },
        "paddle",
    )
    def adjoint(self):
        return torch_frontend.adjoint(self)

    @with_unsupported_dtypes(
        {"2.0.1 and below": ("int16", "float16", "bfloat16")}, "torch"
    )
    def conj(self):
        return torch_frontend.conj(self)

    @with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, "torch")
    def svd(self, some=True, compute_uv=True, *, out=None):
        return torch_frontend.svd(self, some=some, compute_uv=compute_uv, out=out)

    @with_unsupported_dtypes(
        {"2.0.1 and below": ("float16", "bfloat16", "float32", "float64", "complex")},
        "torch",
    )
    def gcd(self, other, *, out=None):
        return torch_frontend.gcd(self, other, out=out)


class Size(tuple):
    def __new__(cls, iterable=()):
        new_iterable = list()
        for i, item in enumerate(iterable):
            if isinstance(item, int):
                new_iterable.append(item)
                continue
            try:
                new_iterable.append(int(item))
            except Exception:
                raise TypeError(f"Expected int, but got {type(item)} at index {i}")
        return super().__new__(cls, new_iterable)

    def __repr__(self):
        return f'ivy.frontends.torch.Size([{", ".join(str(d) for d in self)}])'
