import inspect

from hypothesis import strategies as st
from typing import Callable, Any
from ..function_test import FunctionHandler, num_positional_args_from_dict
from ...globals import TestData
from ...test_parameter_flags import (
    build_function_flags_with_defaults,
)  # TODO should be removed


class BackendFunctionHandler(FunctionHandler):
    def __init__(
        self, fn_tree: str, ground_truth="tensorflow", test_flags=None, **_given_kwargs
    ):
        # Changing the order of init vars will likely break things. Change with caution!
        self.fn_tree = fn_tree
        self.ground_truth = ground_truth
        self._given_kwargs = _given_kwargs
        self.callable_fn = self.import_function(self.fn_tree)
        self._build_test_data()

        if test_flags is None:
            self._build_test_flags_with_defaults()
        else:
            self.test_flags = test_flags

        self._init_possible_args()

    def _init_possible_args(self):
        self.possible_args = {
            "ground_truth_backend": st.just(self.ground_truth),
            "fn_name": st.just(self.test_data.fn_name),
            "test_data": self.test_data,
            "test_flags": self.test_flags,
        }

    def _build_test_data(self):
        module_tree, fn_name = self._partition_fn_tree(self.fn_tree)
        supported_device_dtypes = self._get_supported_devices_dtypes(self.fn_tree)
        self.test_data = TestData(
            module_tree=module_tree,
            fn_name=fn_name,
            supported_device_dtypes=supported_device_dtypes,
        )

    def _build_test_flags_with_defaults(self):
        dict_for_num_pos_strategy = self._build_parameters_info_dict(self.fn_tree)
        self.test_flags = build_function_flags_with_defaults(
            ground_truth_backend=st.just(self.ground_truth),
            num_positional_args=num_positional_args_from_dict(
                dict_for_num_pos_strategy
            ),
        )

    @property
    def is_hypothesis_test(self) -> bool:
        return len(self._given_kwargs.items()) > 0

    def _add_test_attrs_to_fn(self, fn: Callable[..., Any]):
        fn._is_ivy_backend_test = True
        fn.ground_truth_backend = self.ground_truth
        fn.test_data = self.test_data
        return fn

    def _update_given_kwargs(self, fn):
        param_names = inspect.signature(fn).parameters.keys()

        # Check if these arguments are being asked for
        filtered_args = set(param_names).intersection(self.possible_args.keys())
        for key in filtered_args:
            self._given_kwargs[key] = self.possible_args[key]

    def __call__(self, fn: Callable[..., Any]):
        if self.is_hypothesis_test:
            self._update_given_kwargs(fn)
            wrapped_fn = self._wrap_with_hypothesis(fn)
        else:
            wrapped_fn = fn

        self._add_test_attrs_to_fn(wrapped_fn)
        self._wrap_handle_not_implemented(wrapped_fn)
        return wrapped_fn
