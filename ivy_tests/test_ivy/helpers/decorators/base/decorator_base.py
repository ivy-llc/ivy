import functools
import pytest
from abc import ABC, abstractmethod
from hypothesis import given
from typing import Callable, Any


class HandlerBase(ABC):
    @abstractmethod
    def __init__(self, fn_tree: str, test_flags, **_given_kwargs):
        pass

    def _append_ivy_to_fn_tree(self, fn_tree):
        return "ivy." + fn_tree

    @property
    def is_hypothesis_test(self) -> bool:
        return len(self._given_kwargs.items()) > 0

    @abstractmethod
    def __call__(self, func: Callable[..., Any]):
        pass

    def _wrap_with_hypothesis(self, func: Callable[..., Any]):
        return given(**self._given_kwargs)(func)

    def _handle_not_implemented(self, func):
        @functools.wraps(func)
        def wrapped_test(*args, **kwargs):
            try:
                func(*args, **kwargs)
            except Exception as e:
                # A string matching is used instead of actual exception due to
                # exception object in with_backend is different from global Ivy
                if e.__class__.__qualname__ == "IvyNotImplementedException":
                    pytest.skip("Function not implemented in backend.")
                else:
                    raise e

        return wrapped_test
