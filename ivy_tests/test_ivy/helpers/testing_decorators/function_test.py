import importlib
import functools
import pytest
import inspect
import ivy.functional.frontends.numpy as np_frontend #TODO wtf?

from abc import ABC, abstractmethod, abstractproperty
from ivy_tests.test_ivy.helpers.pipeline_helper import update_backend
from hypothesis import given, strategies as st
from ivy_tests.test_ivy.helpers import globals as test_globals
from ivy_tests.test_ivy.helpers.structs import ParametersInfo
from typing import List, Callable, Any
from ivy_tests.test_ivy.helpers.available_frameworks import available_frameworks
from ivy_tests.test_ivy.helpers.hypothesis_helpers.dtype_helpers import (
    _dtype_kind_keys,
    _get_type_dict,
)


@st.composite
def num_positional_args_from_dict(draw, backends_dict):
    parameter_info = backends_dict[test_globals.CURRENT_BACKEND]
    return draw(
        st.integers(
            min_value=parameter_info.positional_only,
            max_value=(parameter_info.total - parameter_info.keyword_only)
        )
    )


class FunctionHandler(ABC):
    @abstractmethod
    def __init__(self, fn_tree: str, test_flags, **_given_kwargs):
        pass

    def import_function(self, fn_tree: str) -> Callable[..., Any]:
        module_tree, _, fn_name = fn_tree.rpartition(".")
        module = importlib.import_module(module_tree)
        return getattr(module, fn_name)

    @abstractproperty
    def is_hypothesis_test(self) -> List[str]:
        pass

    @abstractproperty
    def parameters(self) -> List[str]:
        pass

    @abstractmethod
    def __call__(self, func: Callable[..., Any]):
        pass

    def _wrap_with_hypothesis(self, func: Callable[..., Any]):
        return given(**self._given_kwargs)(func)

    def _wrap_handle_not_implemented(self, func):
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
