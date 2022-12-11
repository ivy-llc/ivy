# local

import ivy
import ivy.functional.frontends.torch as torch_frontend
from ivy.func_wrapper import with_unsupported_dtypes


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
        self.data = self.add(other, alpha=alpha)
        return self.data

    def asin(self):
        return torch_frontend.asin(self.data)

    def asin_(self):
        self.data = self.asin()
        return self.data

    def sin(self):
        return torch_frontend.sin(self.data)

    def sin_(self):
        self.data = self.sin()
        return self.data

    def sinh(self):
        return torch_frontend.sinh(self.data)

    def sinh_(self):
        self.data = self.sinh()
        return self.data

    def cos(self):
        return torch_frontend.cos(self.data)

    def cos_(self):
        self.data = self.cos()
        return self.data

    def cosh(self):
        return torch_frontend.cosh(self.data)

    def cosh_(self):
        self.data = self.cosh()
        return self.data

    def arcsin(self):
        return torch_frontend.arcsin(self.data)

    def arcsin_(self):
        self.data = self.arcsin()
        return self.data

    def atan(self):
        return torch_frontend.atan(self.data)

    def atan_(self):
        self.data = self.atan()
        return self.data

    def view(self, shape):
        self.data = torch_frontend.reshape(self.data, shape)
        return self.data

    def float(self, memory_format=None):
        return ivy.astype(self.data, ivy.float32)

    def asinh(self):
        return torch_frontend.asinh(self.data)

    def asinh_(self):
        self.data = self.asinh()
        return self.data

    def tan(self):
        return torch_frontend.tan(self.data)

    def tan_(self):
        self.data = self.tan()
        return self.data

    def tanh(self):
        return torch_frontend.tanh(self.data)

    def tanh_(self):
        self.data = self.tanh()
        return self.data

    def atanh(self):
        return torch_frontend.atanh(self.data)

    def atanh_(self):
        self.data = self.atanh()
        return self.data

    def arctanh(self):
        return torch_frontend.arctanh(self.data)

    def arctanh_(self):
        self.data = self.arctanh()
        return self.data

    def log(self):
        return ivy.log(self.data)

    def amax(self, dim=None, keepdim=False):
        return torch_frontend.amax(self.data, dim=dim, keepdim=keepdim)

    def amin(self, dim=None, keepdim=False):
        return torch_frontend.amin(self.data, dim=dim, keepdim=keepdim)

    def abs(self):
        return torch_frontend.abs(self.data)

    def abs_(self):
        self.data = self.abs()
        return self.data

    def bitwise_and(self, other):
        return torch_frontend.bitwise_and(self.data, other)

    def contiguous(self, memory_format=None):
        return self.data

    def new_ones(self, size, *, dtype=None, device=None, requires_grad=False):
        return torch_frontend.ones(
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
        self.data = self.arctan()
        return self.data

    def acos(self):
        return torch_frontend.acos(self.data)

    def acos_(self):
        self.data = self.acos()
        return self.data

    def arccos(self):
        return torch_frontend.arccos(self.data)

    def arccos_(self):
        self.data = self.arccos()
        return self.data

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
        self.data = self.unsqueeze(dim)
        return self.data

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
        self.data = self.pow(other)
        return self.data

    def argmax(self, dim=None, keepdim=False):
        return torch_frontend.argmax(self.data, dim=dim, keepdim=keepdim)

    def ceil(self):
        return torch_frontend.ceil(self.data)

    def min(self, dim=None, keepdim=False):
        return torch_frontend.min(self.data, dim=dim, keepdim=keepdim)

    def sqrt(self):
        return torch_frontend.sqrt(self.data)

    def square(self):
        return torch_frontend.square(self.data)

    # Special Methods #
    # -------------------#

    def __add__(self, other, *, alpha=1):
        return torch_frontend.add(self.data, other, alpha=alpha)

    def __mod__(self, other):
        return torch_frontend.remainder(self, other)

    def __long__(self, memory_format=None):
        return Tensor(ivy.astype(self.data, ivy.int64))

    def __getitem__(self, query):
        ret = ivy.get_item(self.data, query)
        return Tensor(ivy.array(ret, dtype=ivy.dtype(ret), copy=False))

    def __radd__(self, other, *, alpha=1):
        return torch_frontend.add(torch_frontend.mul(other, alpha), self, alpha=1)

    def __mul__(self, other):
        return torch_frontend.mul(self, other)

    def __rmul__(self, other):
        return torch_frontend.mul(other, self)

    def __sub__(self, other, *, alpha=1):
        return torch_frontend.subtract(self, other, alpha=alpha)

    def __truediv__(self, other, *, rounding_mode=None):
        return torch_frontend.div(self, other, rounding_mode=rounding_mode)

    # Method aliases
    absolute, absolute_ = abs, abs_
    ndimension = dim


# Tensor (alias)
tensor = Tensor

# ex_tensor = tensor(data=[3, 4])
# print(ex_tensor)
