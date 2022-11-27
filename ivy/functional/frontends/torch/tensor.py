# local
import ivy
import ivy.functional.frontends.torch as torch_frontend
from ivy.func_wrapper import with_unsupported_dtypes
import weakref


class Tensor:
    def __init__(self, array, device=None):
        self._ivyArray = ivy.asarray(array, dtype=torch_frontend.float32, device=device)

    def __repr__(self):
        return (
            "ivy.functional.frontends.torch.Tensor("
            + str(ivy.to_list(self._ivyArray))
            + ")"
        )

    # Properties #
    # ---------- #

    @property
    def ivyArray(self):
        return self._ivyArray

    # Setters #
    # --------#

    @ivyArray.setter
    def ivyArray(self, array):
        self._ivyArray = ivy.array(array) if not isinstance(array, ivy.Array) else array

    # Instance Methods #
    # ---------------- #
    def reshape(self, shape):
        return torch_frontend.reshape(self._ivyArray, shape)

    def add(self, other, *, alpha=1):
        return torch_frontend.add(self._ivyArray, other, alpha=alpha)

    def add_(self, other, *, alpha=1):
        self._ivyArray = self.add(other, alpha=alpha).ivyArray
        return self

    def asin(self):
        return torch_frontend.asin(self._ivyArray)

    def asin_(self):
        self._ivyArray = self.asin().ivyArray
        return self

    def sin(self):
        return torch_frontend.sin(self._ivyArray)

    def sin_(self):
        self._ivyArray = self.sin().ivyArray
        return self

    def sinh(self):
        return torch_frontend.sinh(self._ivyArray)

    def sinh_(self):
        self._ivyArray = self.sinh().ivyArray
        return self

    def cos(self):
        return torch_frontend.cos(self._ivyArray)

    def cos_(self):
        self._ivyArray = self.cos().ivyArray
        return self

    def cosh(self):
        return torch_frontend.cosh(self._ivyArray)

    def cosh_(self):
        self._ivyArray = self.cosh().ivyArray
        return self

    def arcsin(self):
        return torch_frontend.arcsin(self._ivyArray)

    def arcsin_(self):
        self._ivyArray = self.arcsin().ivyArray
        return self

    def atan(self):
        return torch_frontend.atan(self._ivyArray)

    def atan_(self):
        self._ivyArray = self.atan().ivyArray
        return self

    def view(self, shape):
        return torch_frontend.ViewTensor(weakref.ref(self), shape=shape)

    def float(self, memory_format=None):
        return ivy.astype(self._ivyArray, ivy.float32)

    def asinh(self):
        return torch_frontend.asinh(self._ivyArray)

    def asinh_(self):
        self._ivyArray = self.asinh().ivyArray
        return self

    def tan(self):
        return torch_frontend.tan(self._ivyArray)

    def tan_(self):
        self._ivyArray = self.tan().ivyArray
        return self

    def tanh(self):
        return torch_frontend.tanh(self._ivyArray)

    def tanh_(self):
        self._ivyArray = self.tanh().ivyArray
        return self

    def atanh(self):
        return torch_frontend.atanh(self._ivyArray)

    def atanh_(self):
        self._ivyArray = self.atanh().ivyArray
        return self

    def arctanh(self):
        return torch_frontend.arctanh(self._ivyArray)

    def arctanh_(self):
        self._ivyArray = self.arctanh().ivyArray
        return self

    def log(self):
        return ivy.log(self._ivyArray)

    def amax(self, dim=None, keepdim=False):
        return torch_frontend.amax(self._ivyArray, dim=dim, keepdim=keepdim)

    def amin(self, dim=None, keepdim=False):
        return torch_frontend.amin(self._ivyArray, dim=dim, keepdim=keepdim)

    def abs(self):
        return torch_frontend.abs(self._ivyArray)

    def abs_(self):
        self._ivyArray = self.abs().ivyArray
        return self

    def bitwise_and(self, other):
        return torch_frontend.bitwise_and(self._ivyArray, other)

    def contiguous(self, memory_format=None):
        return torch_frontend.tensor(self.ivyArray)

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
                return ivy.asarray(self._ivyArray, dtype=args[0], copy=False)
            else:
                return ivy.asarray(
                    self._ivyArray,
                    dtype=args[0].dtype,
                    device=args[0].device,
                    copy=False,
                )
        else:
            return ivy.asarray(
                self._ivyArray,
                device=kwargs["device"],
                dtype=kwargs["dtype"],
                copy=False,
            )

    def arctan(self):
        return torch_frontend.atan(self._ivyArray)

    @with_unsupported_dtypes({"1.11.0 and below": ("bfloat16")}, "torch")
    def arctan_(self):
        self._ivyArray = self.arctan().ivyArray
        return self

    def acos(self):
        return torch_frontend.acos(self._ivyArray)

    def acos_(self):
        self._ivyArray = self.acos().ivyArray
        return self

    def arccos(self):
        return torch_frontend.arccos(self._ivyArray)

    def arccos_(self):
        self._ivyArray = self.arccos().ivyArray
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
        dtype = ivy.dtype(self._ivyArray) if dtype is None else dtype
        device = ivy.dev(self._ivyArray) if device is None else device
        _data = ivy.asarray(data, copy=True, dtype=dtype, device=device)
        return torch_frontend.tensor(_data)

    def view_as(self, other):
        return self.view(other.shape)

    def expand(self, *sizes):
        return ivy.broadcast_to(self._ivyArray, shape=sizes)

    def detach(self):
        return ivy.stop_gradient(self._ivyArray, preserve_type=False)

    def unsqueeze(self, dim):
        return torch_frontend.unsqueeze(self, dim)

    def unsqueeze_(self, dim):
        self._ivyArray = self.unsqueeze(dim).ivyArray
        return self

    def dim(self):
        return self._ivyArray.ndim

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
        dtype = ivy.dtype(self._ivyArray) if dtype is None else dtype
        device = ivy.dev(self._ivyArray) if device is None else device
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
        dtype = ivy.dtype(self._ivyArray) if dtype is None else dtype
        device = ivy.dev(self._ivyArray) if device is None else device
        _data = ivy.empty(size, dtype=dtype, device=device)
        return torch_frontend.tensor(_data)

    def unfold(self, dimension, size, step):
        slices = []
        for i in range(0, self._ivyArray.shape[dimension] - size + 1, step):
            slices.append(self._ivyArray[i : i + size])
        return ivy.stack(slices)

    def long(self, memory_format=None):
        return ivy.astype(self._ivyArray, ivy.int64)

    def max(self, dim=None, keepdim=False):
        return torch_frontend.max(self._ivyArray, dim=dim, keepdim=keepdim)

    def device(self):
        return ivy.dev(self._ivyArray)

    def is_cuda(self):
        return "gpu" in ivy.dev(self._ivyArray)

    def pow(self, other):
        return ivy.pow(self._ivyArray, other)

    def pow_(self, other):
        self._ivyArray = self.pow(other).ivyArray
        return self

    def size(self, dim=None):
        shape = ivy.shape(self._ivyArray, as_array=True)
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
        return torch_frontend.matmul(self._ivyArray, tensor2)

    def argmax(self, dim=None, keepdim=False):
        return torch_frontend.argmax(self._ivyArray, dim=dim, keepdim=keepdim)

    def ceil(self):
        return torch_frontend.ceil(self._ivyArray)

    def min(self, dim=None, keepdim=False):
        return torch_frontend.min(self._ivyArray, dim=dim, keepdim=keepdim)

    # Special Methods #
    # -------------------#

    def __add__(self, other, *, alpha=1):
        return torch_frontend.add(self._ivyArray, other, alpha=alpha)

    def __mod__(self, other):
        return torch_frontend.remainder(self._ivyArray, other)

    def __long__(self, memory_format=None):
        return torch_frontend.tensor(ivy.astype(self._ivyArray, ivy.int64))

    def __getitem__(self, query):
        ret = ivy.get_item(self._ivyArray, query)
        return torch_frontend.tensor(ivy.array(ret, dtype=ivy.dtype(ret), copy=False))

    def __radd__(self, other, *, alpha=1):
        return torch_frontend.add(
            torch_frontend.mul(other, alpha), self._ivyArray, alpha=1
        )

    def __mul__(self, other):
        return torch_frontend.mul(self._ivyArray, other)

    def __rmul__(self, other):
        return torch_frontend.mul(other, self._ivyArray)

    def __sub__(self, other, *, alpha=1):
        return torch_frontend.subtract(self._ivyArray, other, alpha=alpha)

    def __truediv__(self, other, *, rounding_mode=None):
        return torch_frontend.div(self._ivyArray, other, rounding_mode=rounding_mode)

    def __iadd__(self, other, *, alpha=1):
        self._ivyArray = self.__add__(other, alpha=alpha).ivyArray
        return self

    def __imod__(self, other):
        self._ivyArray = self.__mod__(other).ivyArray
        return self

    def __imul__(self, other):
        self._ivyArray = self.__mul__(other).ivyArray
        return self

    def __isub__(self, other, *, alpha=1):
        self._ivyArray = self.__sub__(other, alpha=alpha).ivyArray
        return self

    def __itruediv__(self, other, *, rounding_mode=None):
        self._ivyArray = self.__truediv__(other, rounding_mode=rounding_mode).ivyArray
        return self

    # Method aliases
    absolute, absolute_ = abs, abs_
    ndimension = dim
