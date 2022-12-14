# local
import ivy
import weakref
import functools
from typing import Callable
from ivy.functional.frontends.torch.tensor import Tensor
import ivy.functional.frontends.torch as torch_frontend


def _merge_from_original(method: Callable) -> Callable:
    @functools.wraps(method)
    def new_method(self, *args, **kwargs):
        if self.ref() is not None:
            self.fetch_from(checked=True)
        return method(self, *args, **kwargs)

    return new_method


def _update_original(self, method, special_method, *args, **kwargs):
    if special_method:
        ret = method(self, *args, **kwargs)
    else:
        ret = method(*args, **kwargs)
    self.merge_to(checked=True)
    return ret


# Decorator for inplace instance methods
def _push_to_original(self, method: Callable) -> Callable:
    @functools.wraps(method)
    def new_method(*args, **kwargs):
        return _update_original(self, method, False, *args, **kwargs)

    return new_method


# Decorator only for inplace special methods
def _merge_to_original(method: Callable) -> Callable:
    @functools.wraps(method)
    def new_method(self, *args, **kwargs):
        if self.ref() is not None:
            return _update_original(self, method, True, *args, **kwargs)
        return method(self, *args, **kwargs)

    return new_method


class ViewTensor:
    def __init__(self, ref, *, shape):
        if isinstance(ref(), Tensor):
            self.delegate = torch_frontend.tensor(
                ivy.reshape(ref().ivy_array, shape, copy=True)
            )
        elif isinstance(ref(), ViewTensor):
            self.delegate = torch_frontend.tensor(
                ivy.reshape(ref().delegate.ivy_array, shape, copy=True)
            )
        else:
            raise TypeError(
                "'ViewTensor' object needs to refer to a 'Tensor' or "
                "'ViewTensor' object"
            )

        self.ref = ref

    def __getattr__(self, item):
        if not hasattr(self.delegate, item):
            raise AttributeError("'Tensor' object has no attribute '{}'".format(item))

        if self.ref() is None:
            return getattr(self.delegate, item)

        self.fetch_from(checked=True)
        attr = getattr(self.delegate, item)
        if callable(attr):
            if len(item) > 1 and item[-1] == "_":
                return _push_to_original(self, attr)

        return attr

    def fetch_from(self, *, checked=False):
        if (self.ref() is not None) or checked:
            if isinstance(self.ref(), Tensor):
                self.delegate = torch_frontend.tensor(
                    ivy.reshape(self.ref().ivy_array, self.size(), copy=True)
                )
            elif isinstance(self.ref(), ViewTensor):
                self.ref().fetch_from()
                self.delegate = torch_frontend.tensor(
                    ivy.reshape(self.ref().delegate.ivy_array, self.size(), copy=True)
                )
            else:
                raise AttributeError(
                    "'ViewTensor' object is not referring to a 'Tensor' or "
                    "'ViewTensor' object"
                )

    def merge_to(self, *, checked=False):
        if (self.ref() is not None) or checked:
            if isinstance(self.ref(), Tensor):
                self.ref().ivy_array = ivy.reshape(
                    self.delegate.ivy_array, self.ref().size(), copy=True
                )
            elif isinstance(self.ref(), ViewTensor):
                self.ref().delegate.ivy_array = ivy.reshape(
                    self.delegate.ivy_array, self.ref().size(), copy=True
                )
                self.ref().merge_to()
            else:
                raise AttributeError(
                    "'ViewTensor' object is not referring to a 'Tensor' or "
                    "'ViewTensor' object"
                )

    # Class Invariance #
    # ---------------- #
    def view(self, shape):
        view = ViewTensor(weakref.ref(self), shape=shape)
        return view

    def size(self):
        return self.delegate.size()

    # Special Methods #
    # --------------- #
    @_merge_from_original
    def __repr__(self):
        return self.delegate.__repr__()

    @_merge_from_original
    def __add__(self, other, *, alpha=1):
        return self.delegate.__add__(other, alpha=alpha)

    @_merge_from_original
    def __mod__(self, other):
        return self.delegate.__mod__(other)

    @_merge_from_original
    def __long__(self, memory_format=None):
        return self.delegate.__long__(memory_format)

    @_merge_from_original
    def __getitem__(self, query):
        return self.delegate.__getitem__(query)

    @_merge_from_original
    def __radd__(self, other, *, alpha=1):
        return self.delegate.__radd__(other, alpha=alpha)

    @_merge_from_original
    def __mul__(self, other):
        return self.delegate.__mul__(other)

    @_merge_from_original
    def __rmul__(self, other):
        return self.delegate.__rmul__(other)

    @_merge_from_original
    def __sub__(self, other, *, alpha=1):
        return self.delegate.__sub__(other, alpha=alpha)

    @_merge_from_original
    def __truediv__(self, other, *, rounding_mode=None):
        return self.delegate.__truediv__(other, rounding_mode=rounding_mode)

    # Inplace Special Methods #
    # ----------------------- #
    @_merge_from_original
    @_merge_to_original
    def __iadd__(self, other, *, alpha=1):
        return self.delegate.__iadd__(other, alpha=alpha)

    @_merge_from_original
    @_merge_to_original
    def __imod__(self, other):
        return self.delegate.__imod__(other)

    @_merge_from_original
    @_merge_to_original
    def __imul__(self, other):
        return self.delegate.__imul__(other)

    @_merge_from_original
    @_merge_to_original
    def __isub__(self, other, *, alpha=1):
        return self.delegate.__isub__(other, alpha=alpha)

    @_merge_from_original
    @_merge_to_original
    def __itruediv__(self, other, *, rounding_mode=None):
        return self.delegate.__itruediv__(other, rounding_mode=rounding_mode)
