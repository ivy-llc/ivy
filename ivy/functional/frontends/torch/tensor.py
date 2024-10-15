# global
from typing import Iterable
import math

# local
import ivy
import ivy.functional.frontends.torch as torch_frontend
from ivy.functional.frontends.numpy.creation_routines.from_existing_data import (
    array as np_frontend_array,
)
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.func_wrapper import with_supported_dtypes
from ivy.func_wrapper import with_supported_device_and_dtypes
from ivy.functional.frontends.torch.func_wrapper import (
    _to_ivy_array,
    numpy_to_torch_style_args,
)


class Tensor:
    def __init__(self, array, device=None, _init_overload=False, requires_grad=False):
        if _init_overload:
            self._ivy_array = (
                array if isinstance(array, ivy.Array) else ivy.array(array)
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

    def __hash__(self):
        return id(self)

    def __setattr__(self, name, value):
        if name == "data":
            self.ivy_array = value.ivy_array
        else:
            super().__setattr__(name, value)

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
        return self.ivy_array.real

    @property
    def imag(self):
        return self.ivy_array.imag

    @property
    def ndim(self):
        return self.dim()

    @property
    def T(self):
        if self.ndim == 1:
            return self
        return torch_frontend.permute(self, list(range(self.ndim))[::-1])

    @property
    def mH(self):
        return torch_frontend.adjoint(self)

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
        return ivy.requires_gradient(self.ivy_array)

    @property
    def is_leaf(self):
        return self._is_leaf

    @property
    def get_device(self):
        if self.device == "cpu":
            return -1
        else:
            return int(self.device.split(":")[-1])

    @property
    def itemsize(self):
        return self.element_size()

    # Setters #
    # --------#

    @device.setter
    def cuda(self, device=None):
        self.device = device
        return self

    @ivy_array.setter
    def ivy_array(self, array):
        self._ivy_array = array if isinstance(array, ivy.Array) else ivy.array(array)

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
            if isinstance(args[0], (tuple, list, ivy.Shape, ivy.NativeShape)):
                shape = args[0]
                return torch_frontend.reshape(self, shape)
            else:
                return torch_frontend.reshape(self, args)
        else:
            raise ValueError("reshape() got no values for argument 'shape'")

    @with_unsupported_dtypes({"2.2 and below": ("bfloat16",)}, "torch")
    @with_unsupported_dtypes({"2.6.0 and below": ("float16",)}, "paddle")
    def reshape_as(self, other):
        return torch_frontend.reshape(self, other.shape)

    @with_unsupported_dtypes({"2.2 and below": ("bfloat16",)}, "torch")
    def add(self, other, *, alpha=1):
        return torch_frontend.add(self, other, alpha=alpha)

    # @with_unsupported_dtypes({"2.2 and below": ("bfloat16",)}, "torch")
    def divide(self, other, *, out=None):
        return torch_frontend.divide(self, other, out=out)

    @with_unsupported_dtypes({"2.2 and below": ("bfloat16",)}, "torch")
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

    @with_unsupported_dtypes({"2.2 and below": ("bfloat16",)}, "torch")
    def add_(self, other, *, alpha=1):
        ret = self.add(other, alpha=alpha)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def addmm(self, mat1, mat2, *, beta=1, alpha=1):
        return torch_frontend.addmm(self, mat1, mat2, beta=beta, alpha=alpha)

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def addmm_(self, mat1, mat2, *, beta=1, alpha=1):
        ret = self.addmm(mat1, mat2, beta=beta, alpha=alpha)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def addmv(self, mat, vec, *, beta=1, alpha=1):
        return torch_frontend.addmv(self, mat, vec, beta=beta, alpha=alpha)

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def addmv_(self, mat, vec, *, beta=1, alpha=1):
        ret = torch_frontend.addmv(self, mat, vec, beta=beta, alpha=alpha)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def addbmm(self, batch1, batch2, *, beta=1, alpha=1):
        return torch_frontend.addbmm(self, batch1, batch2, beta=beta, alpha=alpha)

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def addbmm_(self, batch1, batch2, *, beta=1, alpha=1):
        ret = self.addbmm(batch1, batch2, beta=beta, alpha=alpha)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    @with_unsupported_dtypes({"2.2 and below": ("bfloat16",)}, "torch")
    def subtract_(self, other, *, alpha=1):
        ret = self.sub(other, alpha=alpha)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def asin(self):
        return torch_frontend.asin(self)

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def asin_(self):
        ret = self.asin()
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    def float_power(self, exponent):
        return torch_frontend.float_power(self, exponent)

    @numpy_to_torch_style_args
    @with_unsupported_dtypes({"2.2 and below": ("bfloat16",)}, "torch")
    def sum(self, dim=None, keepdim=False, *, dtype=None):
        return torch_frontend.sum(self, dim=dim, keepdim=keepdim, dtype=dtype)

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def sin(self):
        return torch_frontend.sin(self)

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def sin_(self):
        ret = self.sin()
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def sinh(self):
        return torch_frontend.sinh(self)

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def sinh_(self):
        ret = self.sinh()
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def cos(self):
        return torch_frontend.cos(self)

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def cos_(self):
        ret = self.cos()
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def cosh(self):
        return torch_frontend.cosh(self)

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def cosh_(self):
        ret = self.cosh()
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def atan(self):
        return torch_frontend.atan(self)

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def atan_(self):
        ret = self.atan()
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    @with_unsupported_dtypes({"2.2 and below": ("float16", "bfloat16")}, "torch")
    def atan2(self, other):
        return torch_frontend.atan2(self, other)

    def view(self, *args, size=None):
        """Reshape Tensor.

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
                isinstance(args[0], (tuple, list, ivy.Shape, ivy.NativeShape))
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

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def asinh(self):
        return torch_frontend.asinh(self)

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def asinh_(self):
        ret = self.asinh()
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def tan(self):
        return torch_frontend.tan(self)

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def tan_(self):
        ret = self.tan()
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def tanh(self):
        return torch_frontend.tanh(self)

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def tanh_(self):
        ret = self.tanh()
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def atanh(self):
        return torch_frontend.atanh(self)

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def atanh_(self):
        ret = self.atanh()
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def log(self):
        return torch_frontend.log(self)

    @with_supported_dtypes({"2.2 and below": ("float32", "float64")}, "torch")
    def log2_(self):
        ret = self.log2()
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    @with_unsupported_dtypes({"2.2 and below": ("float16", "bfloat16")}, "torch")
    def logit(self):
        return torch_frontend.logit(self)

    @with_unsupported_dtypes({"2.2 and below": ("bfloat16", "uint16")}, "torch")
    def copy_(self, other, non_blocking=False):
        ret = torch_frontend.tensor(other)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def log_(self):
        ret = self.log()
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def log2(self):
        return torch_frontend.log2(self)

    @with_unsupported_dtypes({"2.2 and below": ("float16", "bfloat16")}, "torch")
    def relu(self):
        return torch_frontend.nn.functional.relu(self)

    @numpy_to_torch_style_args
    @with_unsupported_dtypes({"2.2 and below": ("complex",)}, "torch")
    def amax(self, dim=None, keepdim=False):
        return torch_frontend.amax(self, dim=dim, keepdim=keepdim)

    @numpy_to_torch_style_args
    @with_unsupported_dtypes({"2.2 and below": ("complex",)}, "torch")
    def amin(self, dim=None, keepdim=False):
        return torch_frontend.amin(self, dim=dim, keepdim=keepdim)

    @numpy_to_torch_style_args
    @with_unsupported_dtypes({"2.2 and below": ("complex", "float16")}, "torch")
    def aminmax(self, dim=None, keepdim=False):
        return torch_frontend.aminmax(self, dim=dim, keepdim=keepdim)

    def abs(self):
        return torch_frontend.abs(self)

    def abs_(self):
        ret = self.abs()
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    @with_unsupported_dtypes({"2.2 and below": ("bfloat16",)}, "torch")
    def logical_and(self, other):
        return torch_frontend.logical_and(self, other)

    def logical_not(self, *, out=None):
        return torch_frontend.logical_not(self, out=out)

    def logical_not_(self):
        ret = self.logical_not()
        self.ivy_array = ivy.inplace_update(
            self.ivy_array, ivy.astype(ret.ivy_array, self.dtype)
        )
        return self

    @with_unsupported_dtypes({"2.2 and below": ("bfloat16",)}, "torch")
    def logical_or(self, other):
        return torch_frontend.logical_or(self, other)

    @with_unsupported_dtypes({"2.2 and below": ("bfloat16",)}, "torch")
    def logical_xor(self, other):
        return torch_frontend.logical_xor(self, other)

    def bitwise_not(self):
        return torch_frontend.bitwise_not(self)

    def bitwise_and(self, other):
        return torch_frontend.bitwise_and(self, other)

    @with_supported_dtypes({"2.2 and below": ("integer",)}, "torch")
    def bitwise_or(self, other):
        return torch_frontend.bitwise_or(self, other)

    def bitwise_left_shift(self, other):
        return torch_frontend.bitwise_left_shift(self, other)

    @with_supported_dtypes({"2.2 and below": ("integer",)}, "torch")
    def bitwise_or_(self, other):
        ret = self.bitwise_or(other)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
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
            size = (
                args[0]
                if isinstance(args[0], (tuple, list, ivy.Shape, ivy.NativeShape))
                else args
            )
        return torch_frontend.ones(
            size, dtype=dtype, device=device, requires_grad=requires_grad
        )

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def floor(self, *, out=None):
        return torch_frontend.floor(self)

    @with_unsupported_dtypes(
        {
            "2.2 and below": (
                "bfloat16",
                "uint8",
                "uint32",
                "uint16",
                "uint64",
                "complex128",
                "complex64",
            )
        },
        "torch",
    )
    def not_equal(self, other, *, out=None):
        return torch_frontend.not_equal(self, other, out=out)

    @with_unsupported_dtypes(
        {
            "2.2 and below": (
                "bfloat16",
                "uint8",
                "uint32",
                "uint16",
                "uint64",
                "complex128",
                "complex64",
            )
        },
        "torch",
    )
    def not_equal_(self, other, *, out=None):
        ret = self.not_equal(other)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    def eq(self, other):
        return torch_frontend.eq(self, other)

    def equal(self, other):
        return torch_frontend.equal(self, other)

    @with_unsupported_dtypes({"2.2 and below": ("float16", "complex")}, "torch")
    def erf(self, *, out=None):
        return torch_frontend.erf(self, out=out)

    @with_supported_dtypes(
        {"2.2 and below": ("float32", "float64", "bfloat16")}, "torch"
    )
    def erf_(self, *, out=None):
        ret = self.erf(out=out)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    @with_supported_device_and_dtypes(
        {"2.2 and below": {"cpu": ("float32", "float64")}},
        "torch",
    )
    def erfc(self, *, out=None):
        return torch_frontend.erfc(self, out=out)

    @with_supported_device_and_dtypes(
        {"2.2 and below": {"cpu": ("float32", "float64")}},
        "torch",
    )
    def erfc_(self, *, out=None):
        ret = self.erfc(out=out)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    def new_zeros(
        self,
        *args,
        size=None,
        dtype=None,
        device=None,
        requires_grad=False,
        layout=None,
        pin_memory=False,
    ):
        if size and args:
            raise TypeError("new_zeros() got multiple values for argument 'size'")
        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device
        if size is None:
            size = args[0] if isinstance(args[0], (tuple, list, ivy.Shape)) else args
        return torch_frontend.zeros(
            size=size, dtype=dtype, device=device, requires_grad=requires_grad
        )

    def to(self, *args, **kwargs):
        device = None
        dtype = None

        # look for device and dtype in the args
        for arg in args:
            if hasattr(arg, "ivy_array") or ivy.is_array(arg):
                device = ivy.dev(arg)
                dtype = ivy.dtype(arg)
            elif (
                isinstance(arg, ivy.NativeDtype)
                or isinstance(arg, ivy.Dtype)
                and hasattr(arg, "as_native_dtype")
                or arg in ivy._all_ivy_dtypes_str
            ):
                dtype = arg
            elif isinstance(arg, (ivy.Device, ivy.NativeDevice, str)):
                if isinstance(arg, str) and not isinstance(
                    arg, (ivy.Device, ivy.NativeDevice)
                ):
                    ivy.utils.assertions.check_elem_in_list(
                        arg,
                        [
                            "cpu",
                            "cuda",
                            "mps",
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
                device = arg

        # look for device and dtype in the kwargs
        if "device" in kwargs:
            device = kwargs["device"]
        if "dtype" in kwargs:
            dtype = kwargs["dtype"]

        if (dtype is None or self.dtype == dtype) and (
            device is None or self.device == ivy.as_ivy_dev(device)
        ):
            return self
        else:
            cast_tensor = self.clone()
            cast_tensor.ivy_array = ivy.asarray(
                self.ivy_array,
                dtype=dtype,
                device=device,
            )
            return cast_tensor

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def acos(self):
        return torch_frontend.acos(self)

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def acos_(self):
        ret = self.acos()
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
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

    @with_unsupported_dtypes({"2.2 and below": ("bfloat16",)}, "torch")
    def view_as(self, other):
        return self.view(size=other.shape)

    def expand(self, *args, size=None):
        if args and size:
            raise TypeError("expand() got multiple values for argument 'size'")
        if args:
            if isinstance(args[0], (tuple, list, ivy.Shape, ivy.NativeShape)):
                size = args[0]
            else:
                size = args
        if isinstance(size, (tuple, list)):
            size = tuple(
                s.item() if isinstance(s, torch_frontend.Tensor) else s for s in size
            )
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
        ret = self.detach()
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    def cpu(self):
        return ivy.to_device(self.ivy_array, "cpu")

    def cuda(self):
        return ivy.to_device(self.ivy_array, "gpu:0")

    @with_unsupported_dtypes({"2.2 and below": ("uint16",)}, "torch")
    @numpy_to_torch_style_args
    def unsqueeze(self, dim):
        return torch_frontend.unsqueeze(self, dim)

    @numpy_to_torch_style_args
    def unsqueeze_(self, dim):
        ret = self.unsqueeze(dim)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
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
        self_shape = tuple(self.shape)
        for i in range(0, self_shape[dimension] - size + 1, step):
            slicing = [slice(None)] * len(self.shape)
            slicing[dimension] = slice(i, i + size)
            slices.append(self.ivy_array[tuple(slicing)])
        stacked = torch_frontend.stack(slices, dim=dimension)
        new_shape = list(self.shape)
        num_slices = (self.shape[dimension] - size) // step + 1
        new_shape[dimension] = num_slices
        if dimension == -1:
            new_shape.insert(dimension, size)
        else:
            new_shape.insert(dimension + 1, size)
        reshaped = stacked.reshape(new_shape)
        dims = list(range(len(stacked.shape)))
        dims[-2], dims[-1] = dims[-1], dims[-2]
        return reshaped.permute(*dims)

    def long(self, memory_format=None):
        self.ivy_array = ivy.astype(self.ivy_array, ivy.int64, copy=False)
        return self

    @numpy_to_torch_style_args
    def max(self, dim=None, keepdim=False):
        return torch_frontend.max(self, dim=dim, keepdim=keepdim)

    @with_unsupported_dtypes(
        {
            "2.2 and below": (
                "complex",
                "bfloat16",
                "bool",
                "uint16",
                "uint32",
                "uint64",
            )
        },
        "torch",
    )
    def maximum(self, other, *, out=None):
        return torch_frontend.maximum(self, other=other, out=out)

    @property
    def is_quantized(self):
        return "q" in ivy.dtype(self.ivy_array)

    @property
    def is_cuda(self):
        return "gpu" in ivy.dev(self.ivy_array)

    @property
    def is_meta(self):
        return "meta" in ivy.dev(self.ivy_array)

    @with_unsupported_dtypes({"2.2 and below": ("uint16", "bool")}, "torch")
    def positive(self):
        return torch_frontend.positive(self)

    @with_unsupported_dtypes({"2.2 and below": ("bfloat16",)}, "torch")
    def pow(self, exponent):
        return torch_frontend.pow(self, exponent)

    def unflatten(self, dim, sizes):
        return torch_frontend.unflatten(self, dim, sizes)

    @with_unsupported_dtypes({"2.2 and below": ("bfloat16",)}, "torch")
    def pow_(self, exponent):
        ret = self.pow(exponent)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    def size(self, dim=None):
        shape = self.ivy_array.shape
        if dim is None:
            return shape
        try:
            return shape[dim]
        except IndexError as e:
            raise IndexError(
                f"Dimension out of range (expected to be in range of [{len(shape)},"
                f" {len(shape) - 1}], but got {dim}"
            ) from e

    def matmul(self, other):
        return torch_frontend.matmul(self, other)

    @with_supported_dtypes(
        {"2.2 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
    )
    def matrix_power(self, n, *, out=None):
        return torch_frontend.linalg.matrix_power(self, n, out=out)

    def argwhere(self):
        return torch_frontend.argwhere(self)

    @numpy_to_torch_style_args
    @with_unsupported_dtypes({"2.2 and below": ("complex", "bool")}, "torch")
    def argmax(self, dim=None, keepdim=False):
        return torch_frontend.argmax(self, dim=dim, keepdim=keepdim)

    @numpy_to_torch_style_args
    @with_unsupported_dtypes({"2.2 and below": ("complex",)}, "torch")
    def argmin(self, dim=None, keepdim=False):
        return torch_frontend.argmin(self, dim=dim, keepdim=keepdim)

    @with_unsupported_dtypes({"2.2 and below": ("complex",)}, "torch")
    def argsort(self, dim=-1, descending=False):
        return torch_frontend.argsort(self, dim=dim, descending=descending)

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
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
            if isinstance(args[0], (tuple, list, ivy.Shape, ivy.NativeShape)):
                dims = args[0]
                return torch_frontend.permute(self, dims)
            else:
                return torch_frontend.permute(self, args)
        else:
            raise ValueError("permute() got no values for argument 'dims'")

    @numpy_to_torch_style_args
    @with_unsupported_dtypes({"2.2 and below": ("float16", "bfloat16")}, "torch")
    def mean(self, dim=None, keepdim=False):
        return torch_frontend.mean(self, dim=dim, keepdim=keepdim)

    @with_unsupported_dtypes({"2.2 and below": ("bfloat16",)}, "torch")
    @numpy_to_torch_style_args
    def nanmean(self, dim=None, keepdim=False):
        return torch_frontend.nanmean(self, dim=dim, keepdim=keepdim)

    @with_unsupported_dtypes({"2.2 and below": ("bfloat16",)}, "torch")
    @numpy_to_torch_style_args
    def nansum(self, dim=None, keepdim=False):
        return torch_frontend.nansum(self, dim=dim, keepdim=keepdim)

    @numpy_to_torch_style_args
    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def median(self, dim=None, keepdim=False):
        return torch_frontend.median(self, dim=dim, keepdim=keepdim)

    def transpose(self, dim0, dim1):
        return torch_frontend.transpose(self, dim0=dim0, dim1=dim1)

    def transpose_(self, dim0, dim1):
        ret = self.transpose(dim0, dim1)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    def t(self):
        return torch_frontend.t(self)

    def t_(self):
        ret = self.t()
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    def flatten(self, start_dim=0, end_dim=-1):
        return torch_frontend.flatten(self, start_dim, end_dim)

    @numpy_to_torch_style_args
    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def cumsum(self, dim, *, dtype=None):
        return torch_frontend.cumsum(self, dim, dtype=dtype)

    @numpy_to_torch_style_args
    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def cumsum_(self, dim, *, dtype=None):
        ret = self.cumsum(dim, dtype=dtype)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    @with_unsupported_dtypes({"2.2 and below": ("float16", "bfloat16")}, "torch")
    def inverse(self):
        return torch_frontend.inverse(self)

    @with_unsupported_dtypes({"2.2 and below": ("bool", "bfloat16")}, "torch")
    def neg(self):
        return torch_frontend.negative(self)

    @with_unsupported_dtypes({"2.2 and below": ("bool",)}, "torch")
    def neg_(self):
        ret = torch_frontend.negative(self)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    __neg__ = neg

    @with_unsupported_dtypes({"2.2 and below": ("bool", "bfloat16")}, "torch")
    def negative(self):
        return torch_frontend.negative(self)

    @with_unsupported_dtypes({"2.0.1 and below": ("bool", "bfloat16")}, "torch")
    def negative_(self):
        ret = torch_frontend.negative(self)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

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

    @with_unsupported_dtypes({"2.2 and below": ("bfloat16",)}, "torch")
    def type_as(self, other):
        if self.dtype != other.dtype:
            return torch_frontend.tensor(ivy.astype(self.ivy_array, other.dtype))
        return self

    def byte(self, memory_format=None):
        self.ivy_array = ivy.astype(self.ivy_array, ivy.uint8, copy=False)
        return self

    @numpy_to_torch_style_args
    def squeeze(self, dim=None):
        return torch_frontend.squeeze(self, dim)

    @numpy_to_torch_style_args
    @with_unsupported_dtypes({"2.2 and below": ("uint16",)}, "torch")
    def squeeze_(self, dim=None):
        ret = self.squeeze(dim)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
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
        ret = self.tril(diagonal=diagonal)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    def index_select(self, dim, index):
        return torch_frontend.index_select(self, dim, index)

    @with_unsupported_dtypes({"2.2 and below": ("float16", "complex")}, "torch")
    def clamp(self, min=None, max=None):
        return torch_frontend.clamp(self, min=min, max=max)

    @with_unsupported_dtypes({"2.2 and below": ("float16", "complex")}, "torch")
    def clamp_(self, min=None, max=None):
        ret = self.clamp(min=min, max=max)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    @with_unsupported_dtypes(
        {"2.2 and below": ("bool", "bfloat16", "float16", "complex")}, "torch"
    )
    def clamp_min(self, min=None):
        return torch_frontend.clamp(self, min=min)

    def clamp_min_(self, min=None):
        ret = self.clamp_min(min)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    @with_unsupported_dtypes({"2.2 and below": ("float16", "bfloat16")}, "torch")
    def sqrt(self):
        return torch_frontend.sqrt(self)

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def rsqrt(self):
        return torch_frontend.rsqrt(self)

    @with_unsupported_dtypes({"2.2 and below": ("float16", "bfloat16")}, "torch")
    def rsqrt_(self):
        ret = self.rsqrt()
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    @with_unsupported_dtypes({"2.2 and below": ("float16", "bfloat16")}, "torch")
    def sqrt_(self):
        ret = self.sqrt()
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    def where(self, condition, other):
        return torch_frontend.tensor(torch_frontend.where(condition, self, other))

    def clone(self, memory_format=None):
        return torch_frontend.tensor(ivy.array(self.ivy_array, copy=True))

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def acosh(self):
        return torch_frontend.acosh(self)

    def masked_fill(self, mask, value):
        dtype = ivy.as_native_dtype(self.dtype)
        return torch_frontend.tensor(
            ivy.astype(torch_frontend.where(mask, value, self), dtype)
        )

    def masked_fill_(self, mask, value):
        ret = self.masked_fill(mask, value)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    def masked_select(self, mask):
        return torch_frontend.masked_select(self, mask)

    def masked_scatter(self, mask, source):
        mask = torch_frontend.broadcast_to(mask, self.shape)
        flat_self = torch_frontend.flatten(self.clone())
        flat_mask = torch_frontend.flatten(mask)
        flat_source = torch_frontend.flatten(source)
        indices = torch_frontend.squeeze(torch_frontend.nonzero(flat_mask), -1)
        flat_self[indices] = flat_source[:indices.numel()]
        return flat_self.reshape(self.shape)

    def masked_scatter_(self, mask, source):
        mask = torch_frontend.broadcast_to(mask, self.shape)
        flat_self = torch_frontend.flatten(self.clone())
        flat_mask = torch_frontend.flatten(mask)
        flat_source = torch_frontend.flatten(source)
        indices = torch_frontend.squeeze(torch_frontend.nonzero(flat_mask), -1)
        flat_self[indices] = flat_source[:indices.numel()]
        ret = flat_self.reshape(self.shape)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    @with_unsupported_dtypes({"2.2 and below": ("float16", "bfloat16")}, "torch")
    def index_add_(self, dim, index, source, *, alpha=1):
        ret = torch_frontend.index_add(self, dim, index, source, alpha=alpha)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    @with_unsupported_dtypes({"2.2 and below": ("float16", "bfloat16")}, "torch")
    def index_add(self, dim, index, source, *, alpha=1):
        return torch_frontend.index_add(self.ivy_array, dim, index, source, alpha=alpha)

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def acosh_(self):
        ret = self.acosh()
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    @with_unsupported_dtypes({"2.2 and below": ("bfloat16",)}, "torch")
    def numpy(self):
        return np_frontend_array(self.ivy_array)

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def sigmoid(self):
        return torch_frontend.sigmoid(self)

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def sigmoid_(self):
        ret = self.sigmoid()
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
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
            if isinstance(args[0], (tuple, list, ivy.Shape, ivy.NativeShape)):
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

    @with_supported_dtypes(
        {"2.2 and below": ("float16", "float32", "float64", "bfloat16")}, "torch"
    )
    def reciprocal_(self):
        ret = torch_frontend.reciprocal(self)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    def remainder_(self, other, *, out=None):
        ret = torch_frontend.remainder(self, other, out=out)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    def bitwise_not_(self):
        ret = self.bitwise_not()
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    def bitwise_and_(self, other):
        ret = self.bitwise_and(other)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    @with_unsupported_dtypes({"2.2 and below": ("float16", "bfloat16")}, "torch")
    def atan2_(self, other):
        ret = self.atan2(other)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    @with_unsupported_dtypes({"2.2 and below": ("bfloat16",)}, "torch")
    def fmax(self, other):
        return torch_frontend.fmax(self, other)

    def fmin(self, other):
        return torch_frontend.fmin(self, other)

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def log_softmax(self, dim=None, _stack_level=3, dtype=None):
        return torch_frontend.nn.functional.log_softmax(self, dim=dim, dtype=dtype)

    def isfinite(self):
        return torch_frontend.isfinite(self)

    def msort(self):
        return torch_frontend.msort(self)

    @with_unsupported_dtypes({"2.2 and below": ("float16", "complex")}, "torch")
    def trunc(self):
        return torch_frontend.trunc(self)

    @with_unsupported_dtypes({"2.2 and below": ("float16", "complex")}, "torch")
    def trunc_(self):
        ret = self.trunc()
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    @with_unsupported_dtypes({"2.2 and below": ("float16", "complex")}, "torch")
    def fix(self):
        return torch_frontend.fix(self)

    @with_unsupported_dtypes({"2.2 and below": ("float16", "complex")}, "torch")
    def fix_(self):
        ret = self.fix()
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    def isinf(self):
        return torch_frontend.isinf(self.ivy_array)

    def is_complex(self):
        return torch_frontend.is_complex(self.ivy_array)

    @with_unsupported_dtypes({"2.2 and below": ("uint16", "bfloat16")}, "torch")
    def is_floating_point(self):
        return torch_frontend.is_floating_point(self.ivy_array)

    @with_unsupported_dtypes({"2.2 and below": ("bfloat16",)}, "torch")
    def isreal(self):
        return torch_frontend.isreal(self.ivy_array)

    def addr(self, vec1, vec2, *, beta=1, alpha=1, out=None):
        return torch_frontend.addr(self, vec1, vec2, beta=beta, alpha=alpha, out=out)

    def addr_(self, vec1, vec2, *, beta=1, alpha=1):
        ret = self.addr(vec1, vec2, beta=beta, alpha=alpha)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    @with_unsupported_dtypes({"2.2 and below": ("float16", "bfloat16")}, "torch")
    def dot(self, inp):
        return torch_frontend.dot(self, inp)

    @with_supported_dtypes({"2.2 and below": ("float32", "float64")}, "torch")
    def bernoulli(self, *, generator=None, out=None):
        return torch_frontend.bernoulli(self.ivy_array, generator=generator, out=out)

    @with_supported_dtypes({"2.2 and below": ("float32", "float64")}, "torch")
    def bernoulli_(self, p, *, generator=None, out=None):
        ret = torch_frontend.bernoulli(
            torch_frontend.full(self.shape, p, dtype=torch_frontend.float64),
            generator=generator,
            out=out,
        )
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    def numel(self):
        shape = self.shape
        return int(ivy.astype(ivy.prod(shape), ivy.int64))

    # Special Methods #
    # -------------------#

    def __bool__(self):
        if len(self.shape) == sum(self.shape):
            return self.ivy_array.to_scalar().__bool__()
        raise ValueError(
            "The truth value of an array with more than one element is ambiguous. "
            "Use a.any() or a.all()"
        )

    @with_unsupported_dtypes({"2.2 and below": ("bfloat16",)}, "torch")
    def __add__(self, other):
        return torch_frontend.add(self, other)

    @with_unsupported_dtypes({"2.2 and below": ("bfloat16",)}, "torch")
    def __mod__(self, other):
        return torch_frontend.remainder(self, other)

    @with_unsupported_dtypes({"2.2 and below": ("bfloat16",)}, "torch")
    def __pow__(self, exponent):
        return self.pow(exponent)

    @with_unsupported_dtypes({"2.2 and below": ("bfloat16",)}, "torch")
    def __rpow__(self, other):
        return torch_frontend.pow(other, self)

    def __long__(self, memory_format=None):
        return self.long()

    def __getitem__(self, query, /):
        ivy_args = ivy.nested_map(_to_ivy_array, [self, query])
        ret = ivy.get_item(*ivy_args)
        return torch_frontend.Tensor(ret, _init_overload=True)

    def __setitem__(self, key, value, /):
        key, value = ivy.nested_map(_to_ivy_array, [key, value])
        self.ivy_array[key] = value

    def __iter__(self):
        if self.ndim == 0:
            raise TypeError("iteration over a 0-d tensor not supported")
        for i in range(self.shape[0]):
            yield self[i]

    @with_unsupported_dtypes({"2.2 and below": ("bfloat16",)}, "torch")
    def __radd__(self, other):
        return torch_frontend.add(other, self)

    @with_unsupported_dtypes({"2.2 and below": ("bfloat16",)}, "torch")
    def __mul__(self, other):
        return torch_frontend.mul(self, other)

    @with_unsupported_dtypes({"2.2 and below": "bfloat16"}, "torch")
    def __matmul__(self, other):
        return torch_frontend.matmul(self, other)

    @with_unsupported_dtypes(
        {
            "2.2 and below": (
                "float16",
                "int8",
                "int16",
                "bool",
                "uint8",
            )
        },
        "torch",
    )
    def __rmul__(self, other):
        return torch_frontend.mul(other, self)

    @with_unsupported_dtypes({"2.2 and below": ("bfloat16",)}, "torch")
    def __sub__(self, other):
        return torch_frontend.subtract(self, other)

    def __truediv__(self, other):
        return torch_frontend.div(self, other)

    @with_unsupported_dtypes({"2.2 and below": ("float16", "complex")}, "torch")
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

    @with_unsupported_dtypes({"2.2 and below": ("bfloat16",)}, "torch")
    def __eq__(self, other):
        if isinstance(other, (list, tuple)):
            return False
        return torch_frontend.eq(self, other)

    @with_unsupported_dtypes({"2.2 and below": ("complex",)}, "torch")
    def __gt__(self, other):
        return torch_frontend.greater(self, other)

    @with_unsupported_dtypes({"2.2 and below": ("bfloat16",)}, "torch")
    def __ge__(self, other):
        return torch_frontend.greater_equal(self, other)

    @with_unsupported_dtypes({"2.2 and below": ("bfloat16",)}, "torch")
    def __ne__(self, other):
        if isinstance(other, (list, tuple)):
            return True
        return self.ne(other)

    @with_unsupported_dtypes({"2.2 and below": ("bfloat16",)}, "torch")
    def __rsub__(self, other):
        return torch_frontend.subtract(other, self)

    @with_unsupported_dtypes({"2.2 and below": ("bfloat16",)}, "torch")
    def __lt__(self, other):
        return torch_frontend.less(self, other)

    @with_unsupported_dtypes({"2.2 and below": ("bfloat16",)}, "torch")
    def __le__(self, other):
        return torch_frontend.less_equal(self, other)

    @with_unsupported_dtypes({"2.2 and below": ("bfloat16",)}, "torch")
    def __or__(self, other):
        return torch_frontend.bitwise_or(self, other)

    @with_supported_dtypes({"2.2 and below": ("integer", "bool")}, "torch")
    def __invert__(self):
        return torch_frontend.bitwise_not(self)

    def __and__(self, other):
        return torch_frontend.bitwise_and(self, other)

    def __iand__(self, other):
        self.ivy_array = self.bitwise_and(other).ivy_array
        return self

    def new(self, *shape_list):
        if isinstance(shape_list[0], (list, tuple)):
            shape_list = shape_list[0]
        return torch_frontend.zeros(shape_list, dtype=self.dtype, device=self.device)

    def __array__(self, dtype=None):
        if dtype is None:
            return ivy.to_numpy(self.ivy_array)
        else:
            return ivy.to_numpy(self.ivy_array).astype(dtype, copy=False)

    def __array_wrap__(self, array):
        if array.dtype == bool:
            array = array.astype("uint8")
        return torch_frontend.tensor(array)

    def bitwise_xor(self, other):
        return torch_frontend.bitwise_xor(self, other)

    def bitwise_xor_(self, other):
        ret = self.bitwise_xor(other)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    def item(self):
        if all(dim == 1 for dim in self.shape):
            if ivy.current_backend_str() == "tensorflow":
                import tensorflow as tf

                return tf.squeeze(self.ivy_array.data)
            else:
                return self.ivy_array.to_scalar()
        else:
            raise ValueError(
                "only one element tensors can be converted to Python scalars"
            )

    def element_size(self):
        dtype = ivy.dtype(self.ivy_array)
        return int(ivy.dtype_bits(dtype) // 8)

    @numpy_to_torch_style_args
    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def cumprod(self, dim, dtype=None):
        return torch_frontend.cumprod(self, dim, dtype=dtype)

    @numpy_to_torch_style_args
    def count_nonzero(self, dim):
        return torch_frontend.count_nonzero(self, dim=dim)

    def cov(self, /, *, correction=1, fweights=None, aweights=None):
        return torch_frontend.cov(
            self, correction=correction, fweights=fweights, aweights=aweights
        )

    @with_unsupported_dtypes({"2.2 and below": ("bfloat16", "float16")}, "torch")
    def exp(self):
        return torch_frontend.exp(self)

    @with_supported_dtypes(
        {"2.2 and below": ("bfloat16", "float32", "float64")}, "torch"
    )
    def expm1(self):
        return torch_frontend.expm1(self)

    # remove "bfloat16" from the below decorator after fixing ivy.Array.__repr__ method
    @with_unsupported_dtypes(
        {"2.2 and below": ("bfloat16", "float16", "complex")}, "torch"
    )
    def expm1_(self):
        ret = torch_frontend.expm1(self)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    # fmt: off
    @with_unsupported_dtypes({"2.2 and below": ("int8", "int16", "int32", "int64", "uint8", "bool", "float16",)},"torch",)  # noqa
    def exp_(self):
        ret = self.exp()
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self
    # fmt: on

    def mul(self, other):
        return torch_frontend.mul(self, other)

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def ceil_(self):
        ret = torch_frontend.ceil(self)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    @with_unsupported_dtypes({"2.2 and below": ("bfloat16",)}, "torch")
    def mul_(self, other):
        ret = self.mul(other)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    @with_unsupported_dtypes({"2.2 and below": ("bfloat16", "float16")}, "torch")
    def round(self, *, decimals=0):
        return torch_frontend.round(self, decimals=decimals)

    @with_unsupported_dtypes({"2.2 and below": ("bfloat16", "float16")}, "torch")
    def round_(self, *, decimals=0):
        ret = self.round(decimals=decimals)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    @numpy_to_torch_style_args
    @with_unsupported_dtypes({"2.2 and below": ("float16", "complex")}, "torch")
    def cross(self, other, dim=-1):
        return torch_frontend.cross(self, other, dim=dim)

    @with_unsupported_dtypes({"2.2 and below": ("float16", "bfloat16")}, "torch")
    def det(self):
        return torch_frontend.det(self)

    def reciprocal(self):
        return torch_frontend.reciprocal(self)

    def fill_(self, value):
        ret = torch_frontend.full_like(
            self, value, dtype=self.dtype, device=self.device
        )
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    def nonzero(self, as_tuple=False):
        return torch_frontend.nonzero(self, as_tuple=as_tuple)

    def mm(self, mat2):
        return torch_frontend.mm(self, mat2)

    @with_unsupported_dtypes({"2.2 and below": ("bfloat16", "float16")}, "torch")
    def square(self):
        return torch_frontend.square(self.ivy_array)

    @with_supported_dtypes(
        {
            "2.2 and below": (
                "float16",
                "float32",
                "float64",
                "int16",
                "int32",
                "int64",
                "uint8",
                "int8",
                "complex64",
                "complex128",
            )
        },
        "torch",
    )
    def square_(self):
        ret = torch_frontend.square(self.ivy_array)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def log10(self):
        return torch_frontend.log10(self.ivy_array)

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def log10_(self):
        ret = self.log10()
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    @with_unsupported_dtypes({"2.2 and below": ("uint16",)}, "torch")
    def zero_(self):
        ret = torch_frontend.zeros_like(self)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    def short(self, memory_format=None):
        self.ivy_array = ivy.astype(self.ivy_array, ivy.int16, copy=False)
        return self

    @numpy_to_torch_style_args
    @with_unsupported_dtypes({"2.2 and below": ("float16", "bfloat16")}, "torch")
    def prod(self, dim=None, keepdim=False, *, dtype=None):
        return torch_frontend.prod(self, dim=dim, keepdim=keepdim, dtype=dtype)

    def div(self, other, *, rounding_mode=None):
        return torch_frontend.div(self, other, rounding_mode=rounding_mode)

    def div_(self, other, *, rounding_mode=None):
        ret = self.div(other, rounding_mode=rounding_mode)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    @with_supported_dtypes(
        {"2.2 and below": ("float16", "float32", "float64", "bfloat16")}, "torch"
    )
    def true_divide_(self, other):
        ret = self.div(other, rounding_mode=None)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    def normal_(self, mean=0, std=1, *, generator=None):
        ret = ivy.random_normal(
            mean=mean,
            std=std,
            shape=self.ivy_array.shape,
            dtype=self.dtype,
            device=self.device,
        )
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret)
        return self

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def addcdiv(self, tensor1, tensor2, *, value=1):
        return torch_frontend.addcdiv(self, tensor1, tensor2, value=value)

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def addcmul(self, tensor1, tensor2, *, value=1):
        return torch_frontend.addcmul(self, tensor1, tensor2, value=value)

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def addcmul_(self, tensor1, tensor2, *, value=1):
        ret = self.addcmul(tensor1, tensor2, value=value)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    sign_decorator_dtypes = ("float16", "complex", "bool")

    @with_unsupported_dtypes({"2.2 and below": sign_decorator_dtypes}, "torch")
    def sign(self):
        return torch_frontend.sign(self.ivy_array)

    @with_unsupported_dtypes({"2.2 and below": sign_decorator_dtypes}, "torch")
    def sign_(self):
        ret = self.sign()
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    @numpy_to_torch_style_args
    def std(self, dim=None, unbiased=True, keepdim=False, *, out=None):
        return torch_frontend.std(
            self, dim=dim, unbiased=unbiased, keepdim=keepdim, out=out
        )

    @with_unsupported_dtypes({"2.2 and below": ("float16", "bfloat16")}, "torch")
    def fmod(self, other, *, out=None):
        return torch_frontend.fmod(self, other, out=out)

    @with_unsupported_dtypes({"2.2 and below": ("float16", "bfloat16")}, "torch")
    def fmod_(self, other):
        ret = self.fmod(other)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    def norm(self, p="fro", dim=None, keepdim=False, dtype=None):
        return torch_frontend.norm(self, p=p, dim=dim, keepdim=keepdim, dtype=dtype)

    def tolist(self):
        return self.ivy_array.to_list()

    @with_unsupported_dtypes({"2.2 and below": ("bfloat16",)}, "torch")
    def multiply(self, other, *, out=None):
        return torch_frontend.multiply(self, other, out=out)

    @with_unsupported_dtypes({"2.2 and below": ("bfloat16",)}, "torch")
    def multiply_(self, other, *, out=None):
        ret = torch_frontend.multiply(self, other, out=out)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    @numpy_to_torch_style_args
    @with_unsupported_dtypes({"2.2 and below": ("float16", "complex")}, "torch")
    def topk(self, k, dim=None, largest=True, sorted=True):
        return torch_frontend.topk(self, k, dim=dim, largest=largest, sorted=sorted)

    rshift_dtypes = ("float16", "bfloat16", "float32", "float64", "bool", "complex")

    @with_unsupported_dtypes({"2.2 and below": rshift_dtypes}, "torch")
    def bitwise_right_shift(self, other, *, out=None):
        return torch_frontend.bitwise_right_shift(self.ivy_array, other)

    @with_supported_dtypes(
        {"2.2 and below": ("uint8", "int8", "int32", "int64")}, "torch"
    )
    def bitwise_right_shift_(self, other, *, out=None):
        ret = self.bitwise_right_shift(other, out=out)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    @with_unsupported_dtypes({"2.2 and below": ("float16", "bfloat16")}, "torch")
    def logdet(self):
        chol = torch_frontend.cholesky(self)
        return 2 * torch_frontend.sum(
            torch_frontend.log(torch_frontend.real(torch_frontend.diagonal(chol)))
        )

    @with_unsupported_dtypes({"2.2 and below": ("float16", "bfloat16")}, "torch")
    def copysign(self, other, *, out=None):
        return torch_frontend.copysign(self, other, out=out)

    @with_supported_dtypes(
        {"2.2 and below": ("float16", "float32", "float64")}, "torch"
    )
    def copysign_(self, other, *, out=None):
        ret = self.copysign(other, out=out)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    @with_unsupported_dtypes(
        {"2.2 and below": ("complex", "bfloat16", "bool")}, "torch"
    )
    def greater(self, other, *, out=None):
        return torch_frontend.greater(self, other, out=out)

    @with_unsupported_dtypes({"2.2 and below": ("bfloat16", "bool")}, "torch")
    def greater_(self, other):
        ret = ivy.astype(self.greater(other).ivy_array, self.dtype)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret)
        return self

    @with_unsupported_dtypes(
        {"2.2 and below": ("complex", "bfloat16", "bool")}, "torch"
    )
    def greater_equal(self, other, *, out=None):
        return torch_frontend.greater_equal(self, other, out=out)

    @with_unsupported_dtypes({"2.2 and below": ("bfloat16", "bool")}, "torch")
    def greater_equal_(self, other):
        ret = self.greater_equal(other)
        self.ivy_array = ivy.inplace_update(
            self.ivy_array, ivy.astype(ret.ivy_array, self.dtype)
        )
        return self

    @with_unsupported_dtypes(
        {"2.2 and below": ("complex", "bfloat16", "bool")}, "torch"
    )
    def less(self, other, *, out=None):
        return torch_frontend.less(self, other, out=out)

    @with_unsupported_dtypes({"2.2 and below": ("bfloat16", "bool")}, "torch")
    def less_(self, other):
        ret = self.less(other)
        self.ivy_array = ivy.inplace_update(
            self.ivy_array, ivy.astype(ret.ivy_array, self.dtype)
        )
        return self

    @with_unsupported_dtypes(
        {"2.2 and below": ("complex", "bfloat16", "bool")}, "torch"
    )
    def less_equal(self, other, *, out=None):
        return torch_frontend.less_equal(self, other, out=out)

    @with_unsupported_dtypes({"2.2 and below": ("bfloat16", "bool")}, "torch")
    def less_equal_(self, other):
        ret = self.less_equal(other)
        self.ivy_array = ivy.inplace_update(
            self.ivy_array, ivy.astype(ret.ivy_array, self.dtype)
        )
        return self

    @with_unsupported_dtypes({"2.2 and below": ("bfloat16",)}, "torch")
    def eq_(self, other):
        ret = torch_frontend.eq(self, other)
        self.ivy_array = ivy.inplace_update(
            self.ivy_array, ivy.astype(ret.ivy_array, self.dtype)
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

    @with_supported_dtypes(
        {"2.2 and below": ("float32", "float64", "bfloat16")}, "torch"
    )
    def log1p(self):
        promoted_type = ivy.promote_types(self.dtype, "float32")
        res = torch_frontend.log1p(self)
        return res.to(promoted_type)

    @with_supported_dtypes({"2.2 and below": ("float32", "float64")}, "torch")
    def log1p_(self):
        promoted_type = ivy.promote_types(self.dtype, "float32")
        ret = torch_frontend.log1p(self)
        ret = ret.to(promoted_type)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    def baddbmm(self, batch1, batch2, *, beta=1, alpha=1):
        return torch_frontend.baddbmm(
            self, batch1=batch1, batch2=batch2, beta=beta, alpha=alpha
        )

    def baddbmm_(self, batch1, batch2, *, beta=1, alpha=1):
        ret = torch_frontend.baddbmm(
            self, batch1=batch1, batch2=batch2, beta=beta, alpha=alpha
        )
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    def bmm(self, mat2):
        return torch_frontend.bmm(self, mat2=mat2)

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def floor_(self):
        ret = self.floor()
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    @with_unsupported_dtypes(
        {
            "2.2 and below": (
                "bfloat16",
                "complex",
                "float64",
                "int8",
                "int64",
            )
        },
        "torch",
    )
    def diff(self, n=1, dim=-1, prepend=None, append=None):
        return torch_frontend.diff(self, n=n, dim=dim, prepend=prepend, append=append)

    def diag(self, diagonal=0):
        return torch_frontend.diag(self, diagonal=diagonal)

    @with_unsupported_dtypes({"2.2 and below": ("bfloat16",)}, "torch")
    def diagonal(self, offset=0, dim1=0, dim2=1):
        return torch_frontend.diagonal(self, offset=offset, dim1=dim1, dim2=dim2)

    def gather(self, dim, index):
        return torch_frontend.gather(self, dim=dim, index=index)

    @with_supported_dtypes(
        {"2.2 and below": ("float32", "float64", "int32", "int64")}, "torch"
    )
    def scatter_add_(self, dim, index, src):
        ret = ivy.put_along_axis(self.ivy_array, index, src, dim, mode="sum")
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret)
        return self

    @with_supported_dtypes(
        {"2.2 and below": ("float32", "float64", "int32", "int64")}, "torch"
    )
    def scatter_(self, dim, index, src, *, reduce=None):
        if reduce is None:
            reduce = "replace"
        else:
            mode_mappings = {
                "add": "sum",
                "multiply": "mul",
            }
            reduce = mode_mappings.get(reduce, reduce)
        ret = ivy.put_along_axis(self.ivy_array, index, src, dim, mode=reduce)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret)
        return self

    @with_supported_dtypes(
        {"2.2 and below": ("float32", "float64", "int32", "int64")}, "torch"
    )
    def scatter_reduce_(self, dim, index, src, reduce, *, include_self=True):
        if reduce == "prod":
            reduce = "mul"
        ret = ivy.put_along_axis(self.ivy_array, index, src, dim, mode=reduce)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret)
        return self

    @with_supported_dtypes(
        {"2.2 and below": ("float32", "float64", "int32", "int64")}, "torch"
    )
    def scatter_add(self, dim, index, src):
        return torch_frontend.scatter_add(self, dim, index, src)

    @with_supported_dtypes(
        {"2.2 and below": ("float32", "float64", "int32", "int64")}, "torch"
    )
    def scatter(self, dim, index, src):
        return torch_frontend.scatter_reduce(self, dim, index, src, reduce="replace")

    @with_supported_dtypes(
        {"2.2 and below": ("float32", "float64", "int32", "int64")}, "torch"
    )
    def scatter_reduce(self, dim, index, src, reduce, *, include_self=True):
        return torch_frontend.scatter_reduce(self, dim, index, src, reduce=reduce)

    def take_along_dim(self, indices, dim):
        return torch_frontend.take_along_dim(self, indices=indices, dim=dim)

    def movedim(self, source, destination):
        return torch_frontend.movedim(self, source=source, destination=destination)

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def addcdiv_(self, tensor1, tensor2, *, value=1):
        ret = self.addcdiv(tensor1=tensor1, tensor2=tensor2, value=value)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    @with_supported_dtypes(
        {
            "2.2 and below": (
                "float32",
                "float64",
                "complex32",
                "complex64",
                "complex128",
            )
        },
        "torch",
    )
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
            raise ValueError("apply_ is only supported on cpu tensors")
        ret = callable(self.ivy_array)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret)
        return self

    def requires_grad_(self, requires_grad=True):
        if ivy.requires_gradient(self.ivy_array) and not requires_grad:
            return ivy.stop_gradient(self.ivy_array)
        return self

    def backward(self, gradient=None, retain_graph=None, create_graph=False):
        if gradient is None and int(torch_frontend.numel(self)) > 1:
            raise RuntimeError("grad can be implicitly created only for scalar outputs")
        if self.grad_fn is None and self._grads is None:
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

    @with_unsupported_dtypes({"2.2 and below": ("float16", "bfloat16")}, "torch")
    def logaddexp(self, other):
        return torch_frontend.logaddexp(self, other)

    @with_unsupported_dtypes({"2.2 and below": ("float16",)}, "torch")
    def logaddexp2(self, other):
        self.ivy_array = torch_frontend.logaddexp2(self, other).ivy_array
        return self

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
        {"2.2 and below": ("int16", "float16", "bfloat16")}, "torch"
    )
    def conj(self):
        return torch_frontend.conj(self)

    @with_unsupported_dtypes({"2.2 and below": ("float16", "bfloat16")}, "torch")
    def svd(self, some=True, compute_uv=True, *, out=None):
        return torch_frontend.svd(self, some=some, compute_uv=compute_uv, out=out)

    @with_unsupported_dtypes(
        {"2.2 and below": ("float16", "bfloat16", "float32", "float64", "complex")},
        "torch",
    )
    def gcd(self, other, *, out=None):
        return torch_frontend.gcd(self, other, out=out)

    @with_unsupported_dtypes(
        {
            "2.2 and below": (
                "float16",
                "bfloat16",
                "uint16",
                "bool",
                "complex64",
                "complex128",
            )
        },
        "torch",
    )
    def isnan(self):
        return torch_frontend.isnan(self)

    def char(self):
        self.ivy_array = ivy.asarray(self.ivy_array, dtype=torch_frontend.char)
        return self

    @with_unsupported_dtypes(
        {
            "2.2 and below": (
                "float16",
                "bfloat16",
                "float32",
                "float64",
                "complex",
                "uint8",
                "int8",
            )
        },
        "torch",
    )
    def lcm(self, other, *, out=None):
        return torch_frontend.lcm(self, other, out=out)

    @with_unsupported_dtypes(
        {
            "2.2 and below": (
                "float16",
                "bfloat16",
                "float32",
                "float64",
                "complex",
                "uint8",
                "uint16",
                "uint32",
                "uint64",
                "int8",
            )
        },
        "torch",
    )
    def lcm_(self, other, *, out=None):
        ret = self.lcm(other, out=out)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    @with_unsupported_dtypes(
        {
            "2.2 and below": (
                "bfloat16",
                "int8",
                "uint8",
                "int16",
                "complex128",
                "complex64",
                "bool",
            )
        },
        "torch",
    )
    def triu_(self, diagonal=0):
        ret = torch_frontend.triu(self, diagonal)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    @with_unsupported_dtypes(
        {"2.2 and below": ("float16", "bfloat16")},
        "torch",
    )
    def quantile(self, q, dim=None, keepdim=False, *, interpolation="linear", out=None):
        return torch_frontend.quantile(
            self, q, dim=dim, keepdim=keepdim, interpolation=interpolation, out=out
        )

    @with_unsupported_dtypes(
        {
            "2.2 and below": (
                "int8",
                "int16",
                "uint8",
                "uint16",
                "uint32",
                "uint64",
                "bfloat16",
                "float64",
            )
        },
        "torch",
    )
    def random_(
        self,
        from_=0,
        to=None,
        *,
        generator=None,
    ):
        if to is None:
            if ivy.is_float_dtype(self.ivy_array):
                to = ivy.finfo(self.dtype).max
            else:
                to = ivy.iinfo(self.dtype).max
        ret = ivy.random_uniform(
            low=from_, high=to, shape=self.size(), dtype=self.dtype
        )
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret)
        return self.ivy_array

    @with_unsupported_dtypes(
        {
            "2.2 and below": (
                "integer",
                "unsigned",
                "bfloat16",
                "bool",
                "complex",
            )
        },
        "torch",
    )
    def uniform_(self, from_=0, to=1, *, generator=None):
        ret = ivy.random_uniform(
            low=from_, high=to, shape=self.shape, dtype=self.dtype, seed=generator
        )
        self.ivy_array = ivy.inplace_update(self.ivy_array, ivy.astype(ret, self.dtype))
        return self

    @with_supported_dtypes({"2.2 and below": ("float32", "float64")}, "torch")
    def frac(self, name=None):
        return torch_frontend.frac(self.ivy_array)

    @with_unsupported_dtypes(
        {
            "2.2 and below": (
                "float16",
                "bfloat16",
            )
        },
        "torch",
    )
    def sinc(self):
        return torch_frontend.sinc(self)

    @with_supported_dtypes(
        {
            "2.2 and below": (
                "float32",
                "float64",
                "bfloat16",
            )
        },
        "torch",
    )
    def sinc_(self):
        ret = torch_frontend.sinc(self)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    @with_unsupported_dtypes({"2.2 and below": ("uint8",)}, "torch")
    def index_fill(self, dim, index, value):
        arr = torch_frontend.moveaxis(self, dim, 0)
        arr[ivy.to_list(index)] = value
        arr = torch_frontend.moveaxis(self, 0, dim)
        return arr

    @with_unsupported_dtypes(
        {
            "2.2 and below": (
                "bfloat16",
                "int8",
                "uint8",
                "uint32",
                "uint16",
                "uint64",
                "int16",
                "float16",
                "complex128",
                "complex64",
                "bool",
            )
        },
        "torch",
    )
    def unique_consecutive(self, return_inverse, return_counts, dim):
        return torch_frontend.unique_consecutive(
            self, return_inverse, return_counts, dim
        )

    @with_unsupported_dtypes(
        {
            "2.2 and below": (
                "uint16",
                "uint32",
                "uint64",
                "bfloat16",
                "float16",
                "complex64",
                "complex128",
            )
        },
        "torch",
    )
    def cummax(self, dim):
        return torch_frontend.cummax(self, dim)

    @with_unsupported_dtypes(
        {
            "2.2 and below": (
                "bfloat16",
                "int8",
                "uint8",
                "uint32",
                "uint16",
                "uint64",
                "int16",
                "complex128",
                "complex64",
            )
        },
        "torch",
    )
    def triu(self, diagonal=0):
        return torch_frontend.triu(self, diagonal)

    @with_unsupported_dtypes(
        {"2.2 and below": ("bfloat16",)},
        "torch",
    )
    def xlogy_(self, *, other, out=None):
        ret = torch_frontend.xlogy(self, other, out=out)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret.ivy_array)
        return self

    @with_unsupported_dtypes(
        {
            "2.2 and below": (
                "bfloat16",
                "uint8",
                "uint32",
                "uint16",
                "uint64",
                "complex128",
                "complex64",
            )
        },
        "torch",
    )
    def ne(self, other):
        return self.not_equal(other)

    @with_unsupported_dtypes(
        {
            "2.2 and below": (
                "bfloat16",
                "uint8",
                "uint32",
                "uint16",
                "uint64",
                "complex128",
                "complex64",
            )
        },
        "torch",
    )
    def ne_(self, other):
        return self.not_equal_(other)

    @with_unsupported_dtypes(
        {
            "2.2 and below": (
                "bfloat16",
                "int8",
                "uint8",
                "uint32",
                "uint16",
                "uint64",
                "int16",
                "float16",
                "complex128",
                "complex64",
                "bool",
            )
        },
        "torch",
    )
    def unique(self, sorted=True, return_inverse=False, return_counts=False, dim=None):
        return torch_frontend.unique(self, sorted, return_inverse, return_counts, dim)

    @with_unsupported_dtypes(
        {
            "2.2 and below": (
                "float16",
                "bfloat16",
            )
        },
        "torch",
    )
    def xlogy(self, *, other, out=None):
        return torch_frontend.xlogy(self, other, out=out)

    @with_unsupported_dtypes({"2.2 and below": "complex"}, "torch")
    def minimum(self, other, *, out=None):
        return torch_frontend.minimum(self, other=other, out=out)

    def rad2deg(self, *, out=None):
        return torch_frontend.rad2deg(self, out=out)

    def fill_diagonal_(self, fill_value, wrap=False):
        ret = ivy.fill_diagonal(self.ivy_array, fill_value, wrap=wrap)
        self.ivy_array = ivy.inplace_update(self.ivy_array, ret)
        return self

    @with_supported_dtypes(
        {"2.2 and below": "valid"},
        "torch",
    )
    def corrcoef(self):
        return torch_frontend.corrcoef(self)

    def index_put(self, indices, values, accumulate=False):
        ret = self.clone()
        if accumulate:
            ret[indices[0]] += values
        else:
            ret[indices[0]] = values
        return ret

    def index_put_(self, indices, values, accumulate=False):
        def _set_add(index):
            self[index] += values

        def _set(index):
            self[index] = values

        if accumulate:
            ivy.map(fn=_set_add, unique={"index": indices})
        else:
            ivy.map(fn=_set, unique={"index": indices})

        return self

    @with_unsupported_dtypes({"2.2 and below": ("float16", "complex")}, "torch")
    def erfinv(self, *, out=None):
        return torch_frontend.erfinv(self, out=out)

    @with_unsupported_dtypes({"2.2 and below": ("float16", "complex")}, "torch")
    def erfinv_(self, *, out=None):
        ret = self.erfinv(out=out)
        self.ivy_array = ivy.inplace_update(
            self.ivy_array, ivy.astype(ret.ivy_array, self.dtype)
        )
        return self

    # Method aliases
    absolute, absolute_ = abs, abs_
    clip, clip_ = clamp, clamp_
    ndimension = dim
    subtract = sub
    sub_ = subtract_
    arctan = atan
    arctan_ = atan_
    arctan2 = atan2
    arctan2_ = atan2_
    gt = greater
    gt_ = greater_
    arcsinh = asinh
    arcsinh_ = asinh_
    arcsin = asin
    arcsin_ = asin_
    arctanh = atanh
    arctanh_ = atanh_
    arccosh = acosh
    arccosh_ = acosh_
    arccos = acos
    arccos_ = acos_
    ge = greater_equal
    ge_ = greater_equal_
    lt = less
    lt_ = less_
    le = less_equal
    le_ = less_equal_


class Size(tuple):
    def __new__(cls, iterable=()):
        iterable = ivy.Shape([]) if iterable == () else iterable
        new_iterable = []
        for i, item in enumerate(iterable):
            if isinstance(item, int):
                new_iterable.append(item)
                continue
            try:
                new_iterable.append(int(item))
            except Exception as e:
                raise TypeError(
                    f"Expected int, but got {type(item)} at index {i}"
                ) from e
        return super().__new__(cls, tuple(new_iterable))

    def __init__(self, shape=()) -> None:
        shape = ivy.Shape([]) if shape == () else shape
        native_shape_type = (
            ivy.NativeShape if ivy.NativeShape.__module__ != "ivy" else ivy.Shape
        )
        self._ivy_shape = (
            shape
            if isinstance(shape, (ivy.Shape, native_shape_type))
            else ivy.shape(shape)
        )

    def __repr__(self):
        return f'ivy.frontends.torch.Size([{", ".join(str(d) for d in self)}])'

    @property
    def ivy_shape(self):
        return self._ivy_shape

    def numel(self):
        return int(ivy.astype(ivy.prod(self), ivy.int64))
