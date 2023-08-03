import functools
import pytest
from abc import ABC, abstractmethod
from hypothesis import given
from typing import Callable, Any

from ivy_tests.test_ivy.helpers.hypothesis_helpers.dtype_helpers import (
    _dtype_kind_keys,
    _get_type_dict,
)


class HandlerBase(ABC):
    @abstractmethod
    def __init__(self, fn_tree: str, test_flags, **_given_kwargs):
        pass

    def _append_ivy_to_fn_tree(self, fn_tree):
        return "ivy." + fn_tree

    @property
    def is_hypothesis_test(self) -> bool:
        return len(self._given_kwargs.items()) > 0

    def _partition_dtypes_into_kinds(self, framework: str, dtypes):
        partitioned_dtypes = {}
        for kind in _dtype_kind_keys:
            partitioned_dtypes[kind] = set(
                _get_type_dict(framework, kind)
            ).intersection(dtypes)
        return partitioned_dtypes

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
