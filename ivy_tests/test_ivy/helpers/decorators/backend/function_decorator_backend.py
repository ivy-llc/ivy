import inspect

from hypothesis import strategies as st
from typing import Callable, Any
from ivy_tests.test_ivy.helpers.globals import TestData
from ivy_tests.test_ivy.helpers.decorators.function_decorator_base import (
    FunctionHandler,
)
from ivy_tests.test_ivy.helpers.test_parameter_flags import (
    build_backend_function_flags,
    BuiltInstanceStrategy,
    BuiltAsVariableStrategy,
    BuiltNativeArrayStrategy,
    BuiltGradientStrategy,
    BuiltContainerStrategy,
    BuiltWithOutStrategy,
    BuiltCompileStrategy,
)


class BackendFunctionHandler(FunctionHandler):
    def __init__(
        self,
        fn_tree: str,
        ground_truth_backend="tensorflow",
        num_positional_args=None,
        instance_method=BuiltInstanceStrategy,
        with_out=BuiltWithOutStrategy,
        test_gradients=BuiltGradientStrategy,
        test_compile=BuiltCompileStrategy,
        as_variable=BuiltAsVariableStrategy,
        native_arrays=BuiltNativeArrayStrategy,
        container_flags=BuiltContainerStrategy,
        **_given_kwargs
    ):
        # Changing the order of init vars will likely break things. Change with caution!
        self.fn_tree = self._append_ivy_to_fn_tree(fn_tree)
        self.ground_truth_backend = ground_truth_backend
        self._given_kwargs = _given_kwargs
        self.callable_fn = self.import_function(self.fn_tree)
        self._build_test_data()

        if num_positional_args is None:
            num_positional_args = self._build_num_positional_arguments_strategy()

        self.test_flags = build_backend_function_flags(
            ground_truth_backend=st.just(ground_truth_backend),
            num_positional_args=num_positional_args,
            instance_method=instance_method,
            with_out=with_out,
            test_gradients=test_gradients,
            test_compile=test_compile,
            as_variable=as_variable,
            native_arrays=native_arrays,
            container_flags=container_flags,
        )

        self._init_possible_args()

    def _init_possible_args(self):
        self.possible_args = {
            "ground_truth_backend": st.just(self.ground_truth_backend),
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

    def _add_test_attrs_to_fn(self, fn: Callable[..., Any]):
        fn._is_ivy_backend_test = True
        fn.ground_truth_backend = self.ground_truth_backend
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
        self._handle_not_implemented(wrapped_fn)
        return wrapped_fn
