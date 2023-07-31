import inspect
import functools

from hypothesis import strategies as st
from typing import List, Callable, Any
from ..function_test import FunctionHandler
from ...globals import TestData
from ...test_parameter_flags import (
    build_function_flags_with_defaults,
)  # TODO should be removed


class BackendFunctionHandler(FunctionHandler):
    def __init__(
        self, fn_tree: str, ground_truth="tensorflow", test_flags=None, **_given_kwargs
    ):
        self.fn_tree = fn_tree
        self.ground_truth = ground_truth
        self._given_kwargs = _given_kwargs
        self.callable_fn = self.import_function(self.fn_tree)
        self.test_data = self._build_test_data()

        if test_flags is None:
            self._build_test_flags_with_defaults()
        else:
            self.test_flags = test_flags

    def _build_test_data(self):
        module_tree, _, fn_name = self.fn_tree.rpartition(".")
        return TestData(module_tree=module_tree, fn_name=fn_name)

    def _build_test_flags_with_defaults(self):
        self.test_flags = build_function_flags_with_defaults(
            ground_truth_backend=st.just(self.ground_truth),
            num_positional_args=st.integers(),
        )

    @property
    def is_hypothesis_test(self) -> bool:
        return len(self._given_kwargs.items()) > 0

    @property
    def paremeters(self) -> List[str]:
        return inspect.signature(self._callable_fn).parameters

    def _add_test_attrs_to_fn(self, fn: Callable[..., Any]):
        fn._is_ivy_backend_test = True
        return fn

    def _bind_test_data(self, fn: Callable[..., Any]):
        if "test_data" in inspect.signature(fn).parameters:
            ret = functools.partial(fn, test_data=self.test_data)
        return ret

    def __call__(self, fn: Callable[..., Any]):
        fn = self._bind_test_data(fn)

        if self.is_hypothesis_test:
            wrapped_fn = self._wrap_with_hypothesis(fn)
        else:
            wrapped_fn = fn

        self._add_test_attrs_to_fn(wrapped_fn)
        return wrapped_fn
