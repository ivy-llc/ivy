# global
import weakref

# local
import ivy
import ivy.functional.frontends.torch as torch_frontend
from ivy.func_wrapper import with_unsupported_dtypes


class Tensor:
    def __init__(self, array, device=None):
        self._ivy_array = ivy.asarray(
            array, dtype=torch_frontend.float32, device=device
        )

    def __repr__(self):
        return str(self._ivy_array.__repr__()).replace(
            "ivy.array", "ivy.frontends.torch.Tensor"
        )

    # Properties #
    # ---------- #

    @property
    def ivy_array(self):
        return self._ivy_array

    @property
    def device(self):
        return ivy.dev(self._ivy_array)

    @property
    def dtype(self):
        return self._ivy_array.dtype

    @property
    def shape(self):
        return self._ivy_array.shape

    # Setters #
    # --------#

    @ivy_array.setter
    def ivy_array(self, array):
        self._ivy_array = (
            ivy.array(array) if not isinstance(array, ivy.Array) else array
        )

    # Instance Methods #
    # ---------------- #
    def reshape(self, *args, shape=None):
        if args and shape:
            raise TypeError("reshape() got multiple values for argument 'shape'")
        if shape is not None:
            return torch_frontend.reshape(self._ivy_array, shape)
        if args:
            if isinstance(args[0], tuple):
                shape = args[0]
                return torch_frontend.reshape(self._ivy_array, shape)
            else:
                return torch_frontend.reshape(self._ivy_array, args)
        return torch_frontend.reshape(self._ivy_array)

    @with_unsupported_dtypes({"1.11.0 and below": ("bfloat16",)}, "torch")
    def reshape_as(self, other):
        return torch_frontend.reshape(self, other.shape)

    @with_unsupported_dtypes({"1.11.0 and below": ("bfloat16",)}, "torch")
    def add(self, other, *, alpha=1):
        return torch_frontend.add(self._ivy_array, other, alpha=alpha)

    def chunk(self, chunks, dim=0):
        return torch_frontend.chunk(self._ivy_array, chunks, dim=dim)

    def any(self, dim=None, keepdim=False, *, out=None):
        return torch_frontend.any(self._ivy_array, dim=dim, keepdim=keepdim, out=out)

    def all(self, dim=None, keepdim=False, *, out=None):
        return torch_frontend.all(self._ivy_array, dim=dim, keepdim=keepdim, out=out)

    @with_unsupported_dtypes({"1.11.0 and below": ("bfloat16",)}, "torch")
    def add_(self, other, *, alpha=1):
        self._ivy_array = self.add(other, alpha=alpha).ivy_array
        return self

    @with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
    def asin(self):
        return torch_frontend.asin(self._ivy_array)

    @with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
    def asin_(self):
        self._ivy_array = self.asin().ivy_array
        return self

    @with_unsupported_dtypes({"1.11.0 and below": ("bfloat16",)}, "torch")
    def sum(self):
        return torch_frontend.sum(self._ivy_array)

    @with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
    def sin(self):
        return torch_frontend.sin(self._ivy_array)

    @with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
    def sin_(self):
        self._ivy_array = self.sin().ivy_array
        return self

    @with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
    def sinh(self):
        return torch_frontend.sinh(self._ivy_array)

    @with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
    def sinh_(self):
        self._ivy_array = self.sinh().ivy_array
        return self

    @with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
    def cos(self):
        return torch_frontend.cos(self._ivy_array)

    @with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
    def cos_(self):
        self._ivy_array = self.cos().ivy_array
        return self

    @with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
    def cosh(self):
        return torch_frontend.cosh(self._ivy_array)

    @with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
    def cosh_(self):
        self._ivy_array = self.cosh().ivy_array
        return self

    @with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
    def arcsin(self):
        return torch_frontend.arcsin(self._ivy_array)

    @with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
    def arcsin_(self):
        self._ivy_array = self.arcsin().ivy_array
        return self

    @with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
    def atan(self):
        return torch_frontend.atan(self._ivy_array)

    @with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
    def atan_(self):
        self._ivy_array = self.atan().ivy_array
        return self

    @with_unsupported_dtypes({"1.11.0 and below": ("float16", "bfloat16")}, "torch")
    def atan2(self, other):
        return torch_frontend.atan2(self._ivy_array, other)

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
        size: optional size

        Returns reshaped tensor
        -------
        """
        if size and not args:
            size_tup = size
        elif args and not size:
            if (
                isinstance(args[0], tuple)
                or isinstance(args[0], list)
                or type(args[0]).__name__ == "Size"
            ) and len(args) == 1:
                size_tup = args[0]
            else:
                size_tup = args
        else:
            raise ValueError(
                "View only accepts as argument ints, tuple or list of ints or "
                "the keyword argument size."
            )
        return torch_frontend.ViewTensor(weakref.ref(self), shape=size_tup)

    def float(self, memory_format=None):
        cast_tensor = self.clone()
        cast_tensor.ivy_array = ivy.astype(self._ivy_array, ivy.float32)
        return cast_tensor

    @with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
    def asinh(self):
        return torch_frontend.asinh(self._ivy_array)

    @with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
    def asinh_(self):
        self._ivy_array = self.asinh().ivy_array
        return self

    @with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
    def tan(self):
        return torch_frontend.tan(self._ivy_array)

    @with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
    def tan_(self):
        self._ivy_array = self.tan().ivy_array
        return self

    @with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
    def tanh(self):
        return torch_frontend.tanh(self._ivy_array)

    @with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
    def tanh_(self):
        self._ivy_array = self.tanh().ivy_array
        return self

    @with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
    def atanh(self):
        return torch_frontend.atanh(self._ivy_array)

    @with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
    def atanh_(self):
        self._ivy_array = self.atanh().ivy_array
        return self

    @with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
    def arctanh(self):
        return torch_frontend.arctanh(self._ivy_array)

    @with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
    def arctanh_(self):
        self._ivy_array = self.arctanh().ivy_array
        return self

    @with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
    def log(self):
        return torch_frontend.log(self._ivy_array)

    def amax(self, dim=None, keepdim=False):
        return torch_frontend.amax(self._ivy_array, dim=dim, keepdim=keepdim)

    def amin(self, dim=None, keepdim=False):
        return torch_frontend.amin(self._ivy_array, dim=dim, keepdim=keepdim)

    def abs(self):
        return torch_frontend.abs(self._ivy_array)

    def abs_(self):
        self._ivy_array = self.abs().ivy_array
        return self

    def bitwise_and(self, other):
        return torch_frontend.bitwise_and(self._ivy_array, other)

    def bitwise_or(self, other, *, out=None):
        return torch_frontend.bitwise_or(self._ivy_array, other)

    def contiguous(self, memory_format=None):
        return torch_frontend.tensor(self.ivy_array)

    def new_ones(self, size, *, dtype=None, device=None, requires_grad=False):
        return torch_frontend.ones(
            size, dtype=dtype, device=device, requires_grad=requires_grad
        )

    def new_zeros(self, size, *, dtype=None, device=None, requires_grad=False):
        return torch_frontend.zeros(
            size, dtype=dtype, device=device, requires_grad=requires_grad
        )

    def to(self, *args, **kwargs):
        if len(args) > 0:
            if isinstance(args[0], ivy.Dtype):
                if self.dtype == args[0]:
                    return self
                else:
                    cast_tensor = self.clone()
                    cast_tensor.ivy_array = ivy.asarray(self._ivy_array, dtype=args[0])
                    return cast_tensor
            else:
                if self.dtype == args[0].dtype and self.device == args[0].device:
                    return self
                else:
                    cast_tensor = self.clone()
                    cast_tensor.ivy_array = ivy.asarray(
                        self._ivy_array,
                        dtype=args[0].dtype,
                        device=args[0].device,
                    )
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
                    self._ivy_array,
                    device=kwargs["device"] if "device" in kwargs else self.device,
                    dtype=kwargs["dtype"] if "dtype" in kwargs else self.dtype,
                )
                return cast_tensor

    @with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
    def arctan(self):
        return torch_frontend.atan(self._ivy_array)

    @with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
    def arctan_(self):
        self._ivy_array = self.arctan().ivy_array
        return self

    @with_unsupported_dtypes({"1.11.0 and below": ("float16", "bfloat16")}, "torch")
    def arctan2(self, other):
        return torch_frontend.arctan2(self._ivy_array, other)

    @with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
    def acos(self):
        return torch_frontend.acos(self._ivy_array)

    @with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
    def acos_(self):
        self._ivy_array = self.acos().ivy_array
        return self

    @with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
    def arccos(self):
        return torch_frontend.arccos(self._ivy_array)

    @with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
    def arccos_(self):
        self._ivy_array = self.arccos().ivy_array
        return self

    def new_tensor(
        self,
        data,
        *,
        dtype=None,
        device=None,
        requires_grad=False,
        layout=None,
        pin_memory=False
    ):
        dtype = ivy.dtype(self._ivy_array) if dtype is None else dtype
        device = ivy.dev(self._ivy_array) if device is None else device
        _data = ivy.asarray(data, copy=True, dtype=dtype, device=device)
        return torch_frontend.tensor(_data)

    def view_as(self, other):
        return self.view(other.shape)

    def expand(self, *args, size=None):
        if args and size:
            raise TypeError("reshape() got multiple values for argument 'size'")
        if args:
            if isinstance(args[0], (tuple, list)):
                size = args[0]
            else:
                size = args

        size = list(size)
        for i, dim in enumerate(size):
            if dim < 0:
                size[i] = self.shape[i]

        return torch_frontend.tensor(
            ivy.broadcast_to(self._ivy_array, shape=tuple(size))
        )

    def expand_as(self, other):
        return self.expand(
            ivy.shape(other.ivy_array if isinstance(other, Tensor) else other)
        )

    def detach(self):
        return torch_frontend.tensor(
            ivy.stop_gradient(self._ivy_array, preserve_type=False)
        )

    def unsqueeze(self, dim):
        return torch_frontend.unsqueeze(self, dim)

    def unsqueeze_(self, dim):
        self._ivy_array = self.unsqueeze(dim).ivy_array
        return self

    def split(self, split_size, dim=0):
        return torch_frontend.split(self, split_size, dim)

    def dim(self):
        return self._ivy_array.ndim

    def new_full(
        self,
        size,
        fill_value,
        *,
        dtype=None,
        device=None,
        requires_grad=False,
        layout=None,
        pin_memory=False
    ):
        dtype = ivy.dtype(self._ivy_array) if dtype is None else dtype
        device = ivy.dev(self._ivy_array) if device is None else device
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
        pin_memory=False
    ):
        dtype = ivy.dtype(self._ivy_array) if dtype is None else dtype
        device = ivy.dev(self._ivy_array) if device is None else device
        _data = ivy.empty(size, dtype=dtype, device=device)
        return torch_frontend.tensor(_data)

    def unfold(self, dimension, size, step):
        slices = []
        for i in range(0, self._ivy_array.shape[dimension] - size + 1, step):
            slices.append(self._ivy_array[i : i + size])
        return torch_frontend.stack(slices)

    def long(self, memory_format=None):
        cast_tensor = self.clone()
        cast_tensor.ivy_array = ivy.astype(self._ivy_array, ivy.int64)
        return cast_tensor

    def max(self, dim=None, keepdim=False):
        return torch_frontend.max(self._ivy_array, dim=dim, keepdim=keepdim)

    def is_cuda(self):
        return "gpu" in ivy.dev(self._ivy_array)

    @with_unsupported_dtypes({"1.11.0 and below": ("bfloat16",)}, "torch")
    def pow(self, exponent):
        return torch_frontend.pow(self._ivy_array, exponent)

    @with_unsupported_dtypes({"1.11.0 and below": ("bfloat16",)}, "torch")
    def pow_(self, exponent):
        self._ivy_array = self.pow(exponent).ivy_array
        return self

    def size(self, dim=None):
        shape = ivy.shape(self._ivy_array)
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
        return torch_frontend.matmul(self._ivy_array, other)

    def argwhere(self):
        return torch_frontend.argwhere(self._ivy_array)

    def argmax(self, dim=None, keepdim=False):
        return torch_frontend.argmax(self._ivy_array, dim=dim, keepdim=keepdim)

    def argmin(self, dim=None, keepdim=False):
        return torch_frontend.argmin(self._ivy_array, dim=dim, keepdim=keepdim)

    def argsort(self, dim=-1, descending=False):
        return torch_frontend.argsort(self._ivy_array, dim=dim, descending=descending)

    @with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
    def ceil(self):
        return torch_frontend.ceil(self._ivy_array)

    def min(self, dim=None, keepdim=False):
        return torch_frontend.min(self._ivy_array, dim=dim, keepdim=keepdim)

    def permute(self, *args, dims=None):
        if args and dims:
            raise TypeError("permute() got multiple values for argument 'dims'")
        if dims is not None:
            return torch_frontend.permute(self._ivy_array, dims)
        if args:
            if isinstance(args[0], tuple):
                dims = args[0]
                return torch_frontend.permute(self._ivy_array, dims)
            else:
                return torch_frontend.permute(self._ivy_array, args)
        return torch_frontend.permute(self._ivy_array)

    def mean(self, dim=None, keepdim=False):
        return torch_frontend.mean(self._ivy_array, dim=dim, keepdim=keepdim)

    def transpose(self, dim0, dim1):
        return torch_frontend.transpose(self._ivy_array, dim0=dim0, dim1=dim1)

    def transpose_(self, dim0, dim1):
        self._ivy_array = self.transpose(dim0, dim1).ivy_array
        return self

    def flatten(self, start_dim, end_dim):
        return torch_frontend.flatten(self._ivy_array, start_dim, end_dim)

    @with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
    def cumsum(self, dim, dtype):
        return torch_frontend.cumsum(self._ivy_array, dim, dtype=dtype)

    def inverse(self):
        return torch_frontend.inverse(self._ivy_array)

    def neg(self):
        return torch_frontend.negative(self._ivy_array)

    def int(self, memory_format=None):
        cast_tensor = self.clone()
        cast_tensor.ivy_array = ivy.astype(self._ivy_array, ivy.int32)
        return cast_tensor

    def bool(self, memory_format=None):
        cast_tensor = self.clone()
        cast_tensor.ivy_array = ivy.astype(self._ivy_array, ivy.bool)
        return cast_tensor

    def type(self, dtype=None, non_blocking=False, **kwargs):
        if ivy.exists(dtype):
            self._ivy_array = ivy.astype(self._ivy_array, dtype)
            return self
        else:
            return str(self._ivy_array.dtype)

    def type_as(self, other):
        if self.dtype != other.dtype:
            self._ivy_array = ivy.astype(self._ivy_array, other.dtype)
            return self
        else:
            pass

    def byte(self, memory_format=None):
        cast_tensor = self.clone()
        cast_tensor.ivy_array = ivy.astype(self._ivy_array, ivy.uint8)
        return cast_tensor

    @with_unsupported_dtypes({"1.11.0 and below": ("bfloat16",)}, "torch")
    def ne(self, other):
        return torch_frontend.ne(self._ivy_array, other)

    def squeeze(self, dim):
        return torch_frontend.squeeze(self._ivy_array, dim)

    def flip(self, dims):
        return torch_frontend.flip(self._ivy_array, dims)

    def sort(self, dim=-1, descending=False):
        return torch_frontend.sort(self._ivy_array, dim=dim, descending=descending)

    def tril(self, diagonal=0):
        return torch_frontend.tril(self._ivy_array, diagonal=diagonal)

    def index_select(self, dim, index):
        return torch_frontend.index_select(self._ivy_array, dim, index)

    @with_unsupported_dtypes({"1.11.0 and below": ("bfloat16", "float16")}, "torch")
    def clamp(self, min=None, max=None, *, out=None):
        if min is not None and max is not None and ivy.all(min > max):
            return torch_frontend.tensor(ivy.array(self._ivy_array).full_like(max))
        return torch_frontend.clamp(self._ivy_array, min=min, max=max, out=out)

    @with_unsupported_dtypes({"1.11.0 and below": ("float16", "bfloat16")}, "torch")
    def sqrt(self):
        return torch_frontend.sqrt(self._ivy_array)

    def where(self, condition, other):
        # TODO: replace with torch_frontend.where when it's added
        return torch_frontend.tensor(ivy.where(condition, self._ivy_array, other))

    def clone(self, memory_format=None):
        return torch_frontend.tensor(ivy.array(self._ivy_array, copy=True))

    @with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
    def acosh(self):
        return torch_frontend.acosh(self._ivy_array)

    def real(self):
        return torch_frontend.real(self._ivy_array)

    def masked_fill(self, mask, value):
        # TODO: replace with torch_frontend.where when it's added
        return torch_frontend.tensor(ivy.where(mask, value, self._ivy_array))

    def masked_fill_(self, mask, value):
        self._ivy_array = self.masked_fill(mask, value).ivy_array
        return self

    @with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
    def acosh_(self):
        self._ivy_array = self.acosh().ivy_array
        return self

    @with_unsupported_dtypes({"1.11.0 and below": ("bfloat16",)}, "torch")
    def numpy(self):
        return ivy.to_numpy(self._ivy_array)

    @with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
    def sigmoid(self):
        return torch_frontend.sigmoid(self.ivy_array)

    # Special Methods #
    # -------------------#

    @with_unsupported_dtypes({"1.11.0 and below": ("bfloat16",)}, "torch")
    def __add__(self, other):
        return torch_frontend.add(self._ivy_array, other)

    @with_unsupported_dtypes({"1.11.0 and below": ("bfloat16",)}, "torch")
    def __mod__(self, other):
        return torch_frontend.remainder(self._ivy_array, other)

    def __long__(self, memory_format=None):
        return torch_frontend.tensor(ivy.astype(self._ivy_array, ivy.int64))

    def __getitem__(self, query):
        ret = ivy.get_item(self._ivy_array, query)
        return torch_frontend.tensor(ivy.array(ret, dtype=ivy.dtype(ret), copy=False))

    def __setitem__(self, key, value):
        if hasattr(value, "ivy_array"):
            value = (
                ivy.to_scalar(value.ivy_array)
                if value.shape == ()
                else ivy.to_list(value)
            )
        self._ivy_array[key] = value

    @with_unsupported_dtypes({"1.11.0 and below": ("bfloat16",)}, "torch")
    def __radd__(self, other):
        return torch_frontend.add(other, self._ivy_array)

    @with_unsupported_dtypes({"1.11.0 and below": ("bfloat16",)}, "torch")
    def __mul__(self, other):
        return torch_frontend.mul(self._ivy_array, other)

    @with_unsupported_dtypes({"1.11.0 and below": ("bfloat16",)}, "torch")
    def __rmul__(self, other):
        return torch_frontend.mul(other, self._ivy_array)

    @with_unsupported_dtypes({"1.11.0 and below": ("bfloat16",)}, "torch")
    def __sub__(self, other):
        return torch_frontend.subtract(self._ivy_array, other)

    def __truediv__(self, other):
        return torch_frontend.div(self._ivy_array, other)

    def __iadd__(self, other):
        self._ivy_array = self.__add__(other).ivy_array
        return self

    def __imod__(self, other):
        self._ivy_array = self.__mod__(other).ivy_array
        return self

    def __imul__(self, other):
        self._ivy_array = self.__mul__(other).ivy_array
        return self

    def __isub__(self, other):
        self._ivy_array = self.__sub__(other).ivy_array
        return self

    def __itruediv__(self, other):
        self._ivy_array = self.__truediv__(other).ivy_array
        return self

    @with_unsupported_dtypes({"1.11.0 and below": ("bfloat16",)}, "torch")
    def __eq__(self, other):
        return torch_frontend.equal(self._ivy_array, other)

    @with_unsupported_dtypes({"1.11.0 and below": ("bfloat16",)}, "torch")
    def __gt__(self, other):
        return torch_frontend.greater(self._ivy_array, other)

    @with_unsupported_dtypes({"1.11.0 and below": ("bfloat16",)}, "torch")
    def __ne__(self, other):
        return torch_frontend.ne(self._ivy_array, other)

    @with_unsupported_dtypes({"1.11.0 and below": ("bfloat16",)}, "torch")
    def __rsub__(self, other):
        return torch_frontend.subtract(other, self._ivy_array)

    @with_unsupported_dtypes({"1.11.0 and below": ("bfloat16",)}, "torch")
    def __lt__(self, other):
        return torch_frontend.less(self._ivy_array, other)

    @with_unsupported_dtypes({"1.11.0 and below": ("bfloat16",)}, "torch")
    def __or__(self, other):
        return torch_frontend.bitwise_or(self._ivy_array, other)

    def __invert__(self):
        return torch_frontend.bitwise_not(self._ivy_array)

    # Method aliases
    absolute, absolute_ = abs, abs_
    ndimension = dim
