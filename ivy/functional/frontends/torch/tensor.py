# local
import ivy
import ivy.functional.frontends.torch as torch_frontend
from ivy.func_wrapper import with_unsupported_dtypes
import weakref


class Tensor:
    def __init__(self, data):
        self.data = ivy.array(data) if not isinstance(data, ivy.Array) else data

    def __repr__(self):
        return (
            "ivy.functional.frontends.torch.Tensor(" + str(ivy.to_list(self.data)) + ")"
        )

    # Instance Methods #
    # ---------------- #
    def reshape(self, shape):
        return torch_frontend.reshape(self.data, shape)

    def add(self, other, *, alpha=1):
        return torch_frontend.add(self.data, other, alpha=alpha)

    def add_(self, other, *, alpha=1):
        self.data = self.add(other, alpha=alpha).data
        return self

    def asin(self):
        return torch_frontend.asin(self.data)

    def asin_(self):
        self.data = self.asin().data
        return self

    def sin(self):
        return torch_frontend.sin(self.data)

    def sin_(self):
        self.data = self.sin().data
        return self

    def sinh(self):
        return torch_frontend.sinh(self.data)

    def sinh_(self):
        self.data = self.sinh().data
        return self

    def cos(self):
        return torch_frontend.cos(self.data)

    def cos_(self):
        self.data = self.cos().data
        return self

    def cosh(self):
        return torch_frontend.cosh(self.data)

    def cosh_(self):
        self.data = self.cosh().data
        return self

    def arcsin(self):
        return torch_frontend.arcsin(self.data)

    def arcsin_(self):
        self.data = self.arcsin().data
        return self

    def atan(self):
        return torch_frontend.atan(self.data)

    def atan_(self):
        self.data = self.atan().data
        return self

    def view(self, shape):
        return torch_frontend.ViewTensor(weakref.ref(self), shape=shape)

    def float(self, memory_format=None):
        return ivy.astype(self.data, ivy.float32)

    def asinh(self):
        return torch_frontend.asinh(self.data)

    def asinh_(self):
        self.data = self.asinh().data
        return self

    def tan(self):
        return torch_frontend.tan(self.data)

    def tan_(self):
        self.data = self.tan().data
        return self

    def tanh(self):
        return torch_frontend.tanh(self.data)

    def tanh_(self):
        self.data = self.tanh().data
        return self

    def atanh(self):
        return torch_frontend.atanh(self.data)

    def atanh_(self):
        self.data = self.atanh().data
        return self

    def arctanh(self):
        return torch_frontend.arctanh(self.data)

    def arctanh_(self):
        self.data = self.arctanh().data
        return self

    def log(self):
        return ivy.log(self.data)

    def amax(self, dim=None, keepdim=False):
        return torch_frontend.amax(self.data, dim=dim, keepdim=keepdim)

    def amin(self, dim=None, keepdim=False):
        return torch_frontend.amin(self.data, dim=dim, keepdim=keepdim)

    def abs(self):
        return torch_frontend.abs(self.data)

    def abs_(self):
        self.data = self.abs().data
        return self

    def bitwise_and(self, other):
        return torch_frontend.bitwise_and(self.data, other)

    def contiguous(self, memory_format=None):
        return Tensor(self.data)

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
                return ivy.asarray(self.data, dtype=args[0], copy=False)
            else:
                return ivy.asarray(
                    self.data, dtype=args[0].dtype, device=args[0].device, copy=False
                )
        else:
            return ivy.asarray(
                self.data, device=kwargs["device"], dtype=kwargs["dtype"], copy=False
            )

    def arctan(self):
        return torch_frontend.atan(self.data)

    @with_unsupported_dtypes({"1.11.0 and below": ("bfloat16")}, "torch")
    def arctan_(self):
        self.data = self.arctan().data
        return self

    def acos(self):
        return torch_frontend.acos(self.data)

    def acos_(self):
        self.data = self.acos().data
        return self

    def arccos(self):
        return torch_frontend.arccos(self.data)

    def arccos_(self):
        self.data = self.arccos().data
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
        dtype = ivy.dtype(self.data) if dtype is None else dtype
        device = ivy.dev(self.data) if device is None else device
        _data = ivy.asarray(data, copy=True, dtype=dtype, device=device)
        return Tensor(_data)

    def view_as(self, other):
        return self.view(other.shape)

    def expand(self, *sizes):
        return ivy.broadcast_to(self.data, shape=sizes)

    def detach(self):
        return ivy.stop_gradient(self.data, preserve_type=False)

    def unsqueeze(self, dim):
        return torch_frontend.unsqueeze(self, dim)

    def unsqueeze_(self, dim):
        self.data = self.unsqueeze(dim).data
        return self

    def dim(self):
        return self.data.ndim

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
        dtype = ivy.dtype(self.data) if dtype is None else dtype
        device = ivy.dev(self.data) if device is None else device
        _data = ivy.full(size, fill_value, dtype=dtype, device=device)
        return Tensor(_data)

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
        dtype = ivy.dtype(self.data) if dtype is None else dtype
        device = ivy.dev(self.data) if device is None else device
        _data = ivy.empty(size, dtype=dtype, device=device)
        return Tensor(_data)

    def unfold(self, dimension, size, step):
        slices = []
        for i in range(0, self.data.shape[dimension] - size + 1, step):
            slices.append(self.data[i : i + size])
        return ivy.stack(slices)

    def long(self, memory_format=None):
        return ivy.astype(self.data, ivy.int64)

    def max(self, dim=None, keepdim=False):
        return torch_frontend.max(self.data, dim=dim, keepdim=keepdim)

    def device(self):
        return ivy.dev(self.data)

    def is_cuda(self):
        return "gpu" in ivy.dev(self.data)

    def pow(self, other):
        return ivy.pow(self.data, other)

    def pow_(self, other):
        self.data = self.pow(other).data
        return self

    def size(self, dim=None):
        shape = ivy.shape(self.data, as_array=True)
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

    def matmul(self, tensor2):
        return torch_frontend.matmul(self.data, tensor2)

    def argmax(self, dim=None, keepdim=False):
        return torch_frontend.argmax(self.data, dim=dim, keepdim=keepdim)

    def ceil(self):
        return torch_frontend.ceil(self.data)

    def min(self, dim=None, keepdim=False):
        return torch_frontend.min(self.data, dim=dim, keepdim=keepdim)

    # Special Methods #
    # -------------------#

    def __add__(self, other, *, alpha=1):
        return torch_frontend.add(self.data, other, alpha=alpha)

    def __mod__(self, other):
        return torch_frontend.remainder(self.data, other)

    def __long__(self, memory_format=None):
        return Tensor(ivy.astype(self.data, ivy.int64))

    def __getitem__(self, query):
        ret = ivy.get_item(self.data, query)
        return Tensor(ivy.array(ret, dtype=ivy.dtype(ret), copy=False))

    def __radd__(self, other, *, alpha=1):
        return torch_frontend.add(torch_frontend.mul(other, alpha), self.data, alpha=1)

    def __mul__(self, other):
        return torch_frontend.mul(self.data, other)

    def __rmul__(self, other):
        return torch_frontend.mul(other, self.data)

    def __sub__(self, other, *, alpha=1):
        return torch_frontend.subtract(self.data, other, alpha=alpha)

    def __truediv__(self, other, *, rounding_mode=None):
        return torch_frontend.div(self.data, other, rounding_mode=rounding_mode)

    def __iadd__(self, other, *, alpha=1):
        self.data = self.__add__(other, alpha=alpha).data
        return self

    def __imod__(self, other):
        self.data = self.__mod__(other).data
        return self

    def __imul__(self, other):
        self.data = self.__mul__(other).data
        return self

    def __isub__(self, other, *, alpha=1):
        self.data = self.__sub__(other, alpha=alpha).data
        return self

    def __itruediv__(self, other, *, rounding_mode=None):
        self.data = self.__truediv__(other, rounding_mode=rounding_mode).data
        return self

    # Method aliases
    absolute, absolute_ = abs, abs_
    ndimension = dim
