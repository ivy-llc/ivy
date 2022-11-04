# local

import ivy
import ivy.functional.frontends.torch as torch_frontend


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

    def asin(self, *, out=None):
        return torch_frontend.asin(self.data, out=out)

    def sin(self, *, out=None):
        return torch_frontend.sin(self.data, out=out)

    def sin_(self):
        self.data = self.sin()
        return self.data

    def sinh(self, *, out=None):
        return torch_frontend.sinh(self.data, out=out)

    def sinh_(self):
        self.data = self.sinh()
        return self.data

    def cos(self, *, out=None):
        return torch_frontend.cos(self.data, out=out)

    def cos_(self):
        self.data = self.cos()
        return self.data

    def arcsin(self, *, out=None):
        return torch_frontend.arcsin(self.data, out=out)

    def atan(self, *, out=None):
        return torch_frontend.atan(self.data, out=out)

    def view(self, shape):
        self.data = torch_frontend.reshape(self.data, shape)
        return self.data

    def float(self, memory_format=None):
        return ivy.astype(self.data, ivy.float32)

    def asinh(self, *, out=None):
        return torch_frontend.asinh(self.data, out=out)

    def asinh_(self):
        self.data = self.asinh()
        return self.data

    def tan(self, *, out=None):
        return torch_frontend.tan(self.data, out=out)

    def log(self):
        return ivy.log(self.data)

    def amax(self, dim=None, keepdim=False):
        return torch_frontend.amax(self.data, dim=dim, keepdim=keepdim)

    def amin(self, dim=None, keepdim=False):
        return torch_frontend.amin(self.data, dim=dim, keepdim=keepdim)

    def abs(self, *, out=None):
        return torch_frontend.abs(self.data, out=out)

    def abs_(self):
        self.data = self.abs()
        return self.data

    def contiguous(self, memory_format=None):
        return self.data

    def new_ones(self, size, *, dtype=None, device=None, requires_grad=False):
        return torch_frontend.ones(
            size, dtype=dtype, device=device, requires_grad=requires_grad
        )

    def to(self, *args, **kwargs):
        if len(args) > 0:
            if isinstance(args[0], ivy.Dtype):
                return self._to_with_dtype(*args, **kwargs)
            elif isinstance(args[0], ivy.Device):
                return self._to_with_device(*args, **kwargs)
            else:
                return self._to_with_tensor(*args, **kwargs)

        else:
            if "tensor" not in kwargs:
                return self._to_with_device(**kwargs)
            else:
                return self._to_with_tensor(**kwargs)

    def _to_with_tensor(
        self, tensor, non_blocking=False, copy=False, *, memory_format=None
    ):
        return ivy.asarray(
            self.data, dtype=tensor.dtype, device=tensor.device, copy=copy
        )

    def _to_with_dtype(
        self, dtype, non_blocking=False, copy=False, *, memory_format=None
    ):
        return ivy.asarray(self.data, dtype=dtype, copy=copy)

    def _to_with_device(
        self, device, dtype=None, non_blocking=False, copy=False, *, memory_format=None
    ):
        return ivy.asarray(self.data, device=device, dtype=dtype, copy=copy)

    def arctan(self, *, out=None):
        return torch_frontend.arctan(self, out=out)

    def acos(self, *, out=None):
        return torch_frontend.acos(self.data, out=out)

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
        _data = ivy.variable(_data) if requires_grad else _data
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
        _data = ivy.variable(_data) if requires_grad else _data
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
        _data = ivy.variable(_data) if requires_grad else _data
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

    # Special Methods #
    # -------------------#

    def __add__(self, other, *, alpha=1):
        return torch_frontend.add(self, other, alpha=alpha)

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

    def __mod__(self, other):
        return ivy.remainder(self.data, other)

    # Method aliases
    absolute, absolute_ = abs, abs_
    ndimension = dim


# Tensor (alias)
tensor = Tensor
