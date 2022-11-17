# local
import ivy
import weakref
import functools
from typing import Callable
from ivy.functional.frontends.torch.tensor import Tensor


def _merge_from_original(method: Callable) -> Callable:
    @functools.wraps(method)
    def new_method(self, *args, **kwargs):
        if self.ref() is not None:
            self.fetch_from(checked=True)
        return method(self, *args, **kwargs)

    return new_method


def _update_original(self, method, *args, **kwargs):
    ret = method(*args, **kwargs)
    self.chain_merge_to(checked=True)
    return ret


def _push_to_original(self, method: Callable) -> Callable:
    @functools.wraps(method)
    def new_method(*args, **kwargs):
        return _update_original(self, method, *args, **kwargs)

    return new_method


def _merge_to_original(method: Callable) -> Callable:
    @functools.wraps(method)
    def new_method(self, *args, **kwargs):
        if self.ref() is not None:
            return _update_original(self, method, *args, **kwargs)
        return method(self, *args, **kwargs)

    return new_method


class ViewTensor:
    def __init__(self, ref, *, shape):
        if isinstance(ref(), Tensor):
            self.delegate = Tensor(ivy.reshape(ref().data, shape, copy=True))
        elif isinstance(ref(), ViewTensor):
            self.delegate = Tensor(ivy.reshape(ref().delegate.data, shape, copy=True))
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
                self.delegate = Tensor(
                    ivy.reshape(self.ref().data, self.size(), copy=True)
                )
            elif isinstance(self.ref(), ViewTensor):
                self.ref().fetch_from()
                self.delegate = Tensor(
                    ivy.reshape(self.ref().delegate.data, self.size(), copy=True)
                )
            else:
                raise AttributeError(
                    "'ViewTensor' object is not referring to a 'Tensor' or "
                    "'ViewTensor' object"
                )

    def chain_merge_to(self, *, checked=False):
        if (self.ref() is not None) or checked:
            if isinstance(self.ref(), Tensor):
                self.ref().data = ivy.reshape(
                    self.delegate.data, self.ref().size(), copy=True
                )
            elif isinstance(self.ref(), ViewTensor):
                self.ref().delegate.data = ivy.reshape(
                    self.delegate.data, self.ref().size(), copy=True
                )
                self.ref().chain_merge_to()
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
