# local
import ivy
import ivy.functional.frontends.torch as torch_frontend
from ivy.func_wrapper import with_unsupported_dtypes
import weakref


class Tensor:
    def __init__(self, array, device=None):
        self._ivy_array = ivy.asarray(
            array, dtype=torch_frontend.float32, device=device
        )

    def __repr__(self):
        return "ivy.frontends.torch.Tensor(" + str(ivy.to_list(self._ivy_array)) + ")"

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

    def add(self, other, *, alpha=1):
        return torch_frontend.add(self._ivy_array, other, alpha=alpha)

    def chunk(self, chunks, dim=0):
        return torch_frontend.chunk(self._ivy_array, chunks, dim=dim)

    def add_(self, other, *, alpha=1):
        self._ivy_array = self.add(other, alpha=alpha).ivy_array
        return self

    def asin(self):
        return torch_frontend.asin(self._ivy_array)

    def asin_(self):
        self._ivy_array = self.asin().ivy_array
        return self

    def sum(self):
        return torch_frontend.sum(self._ivy_array)

    def sin(self):
        return torch_frontend.sin(self._ivy_array)

    def sin_(self):
        self._ivy_array = self.sin().ivy_array
        return self

    def sinh(self):
        return torch_frontend.sinh(self._ivy_array)

    def sinh_(self):
        self._ivy_array = self.sinh().ivy_array
        return self

    def cos(self):
        return torch_frontend.cos(self._ivy_array)

    def cos_(self):
        self._ivy_array = self.cos().ivy_array
        return self

    def cosh(self):
        return torch_frontend.cosh(self._ivy_array)

    def cosh_(self):
        self._ivy_array = self.cosh().ivy_array
        return self

    def arcsin(self):
        return torch_frontend.arcsin(self._ivy_array)

    def arcsin_(self):
        self._ivy_array = self.arcsin().ivy_array
        return self

    def atan(self):
        return torch_frontend.atan(self._ivy_array)

    def atan_(self):
        self._ivy_array = self.atan().ivy_array
        return self

    def view(self, shape):
        return torch_frontend.ViewTensor(weakref.ref(self), shape=shape)

    def float(self, memory_format=None):
        return ivy.astype(self._ivy_array, ivy.float32)

    def asinh(self):
        return torch_frontend.asinh(self._ivy_array)

    def asinh_(self):
        self._ivy_array = self.asinh().ivy_array
        return self

    def tan(self):
        return torch_frontend.tan(self._ivy_array)

    def tan_(self):
        self._ivy_array = self.tan().ivy_array
        return self

    def tanh(self):
        return torch_frontend.tanh(self._ivy_array)

    def tanh_(self):
        self._ivy_array = self.tanh().ivy_array
        return self

    def atanh(self):
        return torch_frontend.atanh(self._ivy_array)

    def atanh_(self):
        self._ivy_array = self.atanh().ivy_array
        return self

    def arctanh(self):
        return torch_frontend.arctanh(self._ivy_array)

    def arctanh_(self):
        self._ivy_array = self.arctanh().ivy_array
        return self

    def log(self):
        return ivy.log(self._ivy_array)

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
                return ivy.asarray(self._ivy_array, dtype=args[0], copy=False)
            else:
                return ivy.asarray(
                    self._ivy_array,
                    dtype=args[0].dtype,
                    device=args[0].device,
                    copy=False,
                )
        else:
            return ivy.asarray(
                self._ivy_array,
                device=kwargs["device"],
                dtype=kwargs["dtype"],
                copy=False,
            )

    def arctan(self):
        return torch_frontend.atan(self._ivy_array)

    @with_unsupported_dtypes({"1.11.0 and below": ("bfloat16")}, "torch")
    def arctan_(self):
        self._ivy_array = self.arctan().ivy_array
        return self

    def acos(self):
        return torch_frontend.acos(self._ivy_array)

    def acos_(self):
        self._ivy_array = self.acos().ivy_array
        return self

    def arccos(self):
        return torch_frontend.arccos(self._ivy_array)

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

    def expand(self, *sizes):
        return ivy.broadcast_to(self._ivy_array, shape=sizes)

    def detach(self):
        return ivy.stop_gradient(self._ivy_array, preserve_type=False)

    def unsqueeze(self, dim):
        return torch_frontend.unsqueeze(self, dim)

    def unsqueeze_(self, dim):
        self._ivy_array = self.unsqueeze(dim).ivy_array
        return self

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
        return ivy.stack(slices)

    def long(self, memory_format=None):
        return ivy.astype(self._ivy_array, ivy.int64)

    def max(self, dim=None, keepdim=False):
        return torch_frontend.max(self._ivy_array, dim=dim, keepdim=keepdim)

    def is_cuda(self):
        return "gpu" in ivy.dev(self._ivy_array)

    def pow(self, exponent):
        return ivy.pow(self._ivy_array, exponent)

    def pow_(self, exponent):
        self._ivy_array = self.pow(exponent).data
        return self

    def size(self, dim=None):
        shape = ivy.shape(self._ivy_array, as_array=True)
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

    def argmax(self, dim=None, keepdim=False):
        return torch_frontend.argmax(self._ivy_array, dim=dim, keepdim=keepdim)

    def argmin(self, dim=None, keepdim=False):
        return torch_frontend.argmin(self._ivy_array, dim=dim, keepdim=keepdim)

    def ceil(self):
        return torch_frontend.ceil(self._ivy_array)

    def min(self, dim=None, keepdim=False):
        return torch_frontend.min(self._ivy_array, dim=dim, keepdim=keepdim)

    def permute(self, dims):
        return torch_frontend.permute(self, dims)

    def mean(self, dim=None, keepdim=False):
        return torch_frontend.mean(self._ivy_array, dim=dim, keepdim=keepdim)

    def transpose(self, dim0, dim1):
        return torch_frontend.transpose(self._ivy_array, dim0=dim0, dim1=dim1)

    def transpose_(self, dim0, dim1):
        self._ivy_array = self.transpose(dim0, dim1).ivy_array
        return self

    def flatten(self, start_dim, end_dim):
        return torch_frontend.flatten(self._ivy_array, start_dim, end_dim)

    def cumsum(self, dim, dtype):
        return torch_frontend.cumsum(self._ivy_array, dim, dtype=dtype)

    def inverse(self):
        return torch_frontend.inverse(self._ivy_array)

    def neg(self):
        return torch_frontend.negative(self._ivy_array)

    def int(self, memory_format=None):
        return ivy.astype(self._ivy_array, ivy.int32)

    def bool(self, memory_format=None):
        return ivy.astype(self._ivy_array, ivy.bool)

    def type(self, dtype=None, non_blocking=False, **kwargs):
        if ivy.exists(dtype):
            return ivy.astype(self._ivy_array, dtype)
        else:
            return str(self._ivy_array.dtype)

    def type_as(self, other):
        if self.dtype == other.dtype:
            return self._ivy_array
        else:
            return ivy.astype(self._ivy_array, other.dtype)

    def byte(self, memory_format=None):
        return ivy.astype(self._ivy_array, ivy.uint8)

    def ne(self, other):
        return torch_frontend.ne(self._ivy_array, other)

    def squeeze(self, dim):
        return torch_frontend.squeeze(self._ivy_array, dim)

    def flip(self, dims):
        return torch_frontend.flip(self._ivy_array, dims)

    def tril(self, diagonal=0):
        return torch_frontend.tril(self._ivy_array, diagonal=diagonal)

    def index_select(self, dim, index):
        return torch_frontend.index_select(self._ivy_array, dim, index)

    def where(self, condition, other):
        return ivy.where(condition, self._ivy_array, other)

    def clone(self, memory_format=None):
        return torch_frontend.tensor(ivy.array(self._ivy_array, copy=True))

    # Special Methods #
    # -------------------#

    def __add__(self, other, *, alpha=1):
        return torch_frontend.add(self._ivy_array, other, alpha=alpha)

    def __mod__(self, other):
        return torch_frontend.remainder(self._ivy_array, other)

    def __long__(self, memory_format=None):
        return torch_frontend.tensor(ivy.astype(self._ivy_array, ivy.int64))

    def __getitem__(self, query):
        ret = ivy.get_item(self._ivy_array, query)
        return torch_frontend.tensor(ivy.array(ret, dtype=ivy.dtype(ret), copy=False))

    def __radd__(self, other, *, alpha=1):
        return torch_frontend.add(
            torch_frontend.mul(other, alpha), self._ivy_array, alpha=1
        )

    def __mul__(self, other):
        return torch_frontend.mul(self._ivy_array, other)

    def __rmul__(self, other):
        return torch_frontend.mul(other, self._ivy_array)

    def __sub__(self, other, *, alpha=1):
        return torch_frontend.subtract(self._ivy_array, other, alpha=alpha)

    def __truediv__(self, other, *, rounding_mode=None):
        return torch_frontend.div(self._ivy_array, other, rounding_mode=rounding_mode)

    def __iadd__(self, other, *, alpha=1):
        self._ivy_array = self.__add__(other, alpha=alpha).ivy_array
        return self

    def __imod__(self, other):
        self._ivy_array = self.__mod__(other).ivy_array
        return self

    def __imul__(self, other):
        self._ivy_array = self.__mul__(other).ivy_array
        return self

    def __isub__(self, other, *, alpha=1):
        self._ivy_array = self.__sub__(other, alpha=alpha).ivy_array
        return self

    def __itruediv__(self, other, *, rounding_mode=None):
        self._ivy_array = self.__truediv__(other, rounding_mode=rounding_mode).ivy_array
        return self

    def __eq__(self, other):
        return ivy.equal(self._ivy_array, other)

    def __gt__(self, other):
        return torch_frontend.greater(self._ivy_array, other)

    def __ne__(self, other):
        return torch_frontend.ne(self._ivy_array, other)

    def __lt__(self, other):
        return torch_frontend.less(self._ivy_array, other)

    def __or__(self, other):
        return torch_frontend.bitwise_or(self._ivy_array, other)

    # Method aliases
    absolute, absolute_ = abs, abs_
    ndimension = dim
