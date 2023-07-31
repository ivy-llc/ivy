import importlib
import inspect
from ..test_parameter_flags import (
    build_function_flags_with_defaults,
)  # TODO should be removed
from abc import ABC, abstractmethod
from hypothesis import given, strategies as st
from typing import List, Callable, Any


class FunctionHandler(ABC):
    @abstractmethod
    def __init__(self, fn_tree: str, test_flags, **_given_kwargs):
        pass

    def import_function(self, fn_tree: str) -> Callable[..., Any]:
        module_tree, _, fn_name = fn_tree.rpartition(".")
        module = importlib.import_module(module_tree)
        return getattr(module, fn_name)

    @abstractmethod
    def paremeters(self) -> List[str]:
        pass

    @abstractmethod
    def __call__(self, func: Callable[..., Any]):
        pass

    def _wrap_with_hypothesis(self, func: Callable[..., Any]):
        return given(test_flags=self.test_flags, **self._given_kwargs)(func)


class BackendFunctionHandler(FunctionHandler):
    def __init__(
        self, fn_tree: str, ground_truth: str, test_flags=None, **_given_kwargs
    ):
        self.fn_tree = fn_tree
        self.ground_truth = ground_truth
        self._callable_fn = self.import_function(self.fn_tree)
        self._given_kwargs = _given_kwargs
        if test_flags is None:
            self._build_test_flags_with_defaults()
        else:
            self.test_flags = test_flags

    def _build_test_flags_with_defaults(self):
        self.test_flags = build_function_flags_with_defaults(
            ground_truth_backend=st.just(self.ground_truth),
            num_positional_args=st.integers(),
        )

    def paremeters(self) -> List[str]:
        return inspect.signature(self._callable_fn).parameters

    def __call__(self, func: Callable[..., Any]):
        # @functools.wraps(func)
        # def new_fn(*args, **kwargs):
        #     return

        return self._wrap_with_hypothesis(func)
