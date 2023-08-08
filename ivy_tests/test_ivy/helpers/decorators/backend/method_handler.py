from hypothesis import strategies as st
from typing import Any, Callable
from ivy_tests.test_ivy.helpers.decorators.base.method_handler_base import (
    MethodHandlerBase,
)
from ivy_tests.test_ivy.helpers.decorators.parameter_info_builder import (
    ParameterInfoStrategyBuilder,
)
from ivy_tests.test_ivy.helpers.structs import SupportedDevicesDtypes
from ivy_tests.test_ivy.helpers.test_parameter_flags import (
    BuiltAsVariableStrategy,
    BuiltNativeArrayStrategy,
    BuiltGradientStrategy,
    BuiltContainerStrategy,
    BuiltCompileStrategy,
    build_backend_method_flags,
    build_init_method_backend_flags,
)


class HandleBackendMethod(MethodHandlerBase):
    IVY_PREFIX = "ivy"

    def __init__(
        self,
        init_tree: str,
        method_tree: str,
        ground_truth_backend: str = "tensorflow",
        test_gradients=BuiltGradientStrategy,  # TODO both test_gradients and
        test_compile=BuiltCompileStrategy,  # test_compile should be hidden in a DTO
        init_num_positional_args=None,
        init_native_arrays=BuiltNativeArrayStrategy,
        init_as_variable_flags=BuiltAsVariableStrategy,
        method_num_positional_args=None,
        method_native_arrays=BuiltNativeArrayStrategy,
        method_as_variable_flags=BuiltAsVariableStrategy,
        method_container_flags=BuiltContainerStrategy,
        **_given_kwargs,
    ):
        self._init_tree = init_tree
        self._method_tree = self._append_ivy_prefix_to_tree(method_tree)
        self.ground_truth_backend = ground_truth_backend
        self.test_compile = test_compile
        self.test_gradients = test_gradients
        self._given_kwargs = _given_kwargs

        if init_num_positional_args is None:
            init_strategy_builder = ParameterInfoStrategyBuilder.from_function(
                self.init_tree
            )
            init_num_positional_args = init_strategy_builder.build()

        self.init_flags = self._build_init_flags(
            init_num_positional_args,
            init_as_variable_flags,
            init_native_arrays,
        )

        if method_num_positional_args is None:
            method_strategy_builder = ParameterInfoStrategyBuilder.from_method(
                self._method_tree
            )
            method_num_positional_args = method_strategy_builder.build()

        self.method_flags = self._build_method_flags(
            method_num_positional_args,
            method_native_arrays,
            method_as_variable_flags,
            method_container_flags,
        )

    @staticmethod
    def _build_init_flags(
        init_num_positional_args, init_as_variable_flags, init_native_arrays
    ):
        return build_init_method_backend_flags(
            num_positional_args=init_num_positional_args,
            as_variable=init_as_variable_flags,
            native_arrays=init_native_arrays,
        )

    @staticmethod
    def _build_method_flags(
        method_num_positional_args,
        method_as_variable_flags,
        method_native_arrays,
        method_container_flags,
    ):
        return build_backend_method_flags(
            num_positional_args=method_num_positional_args,
            as_variable=method_as_variable_flags,
            native_arrays=method_native_arrays,
            container_flags=method_container_flags,
        )

    @property
    def init_tree(self):
        return self._init_tree

    @property
    def method_tree(self):
        return self._method_tree

    @property
    def possible_arguments(self):
        class_tree_name, _, method_name = self.method_tree.rpartition(".")
        _, _, class_name = class_tree_name.rpartition(".")
        return {
            "class_name": st.just(class_name),
            "method_name": st.just(method_name),
            "ground_truth_backend": st.just(self.ground_truth_backend),
            "init_flags": self.init_flags,
            "method_flags": self.method_flags,
            "test_compile": self.test_compile,
            "test_gradients": self.test_gradients,
        }

    @property
    def given_kwargs(self):
        return self._given_kwargs

    @property
    def test_data(self):
        return SupportedDevicesDtypes(self.supported_devices_dtypes)

    def _add_test_attributes_to_test_function(self, test_function: Callable[..., Any]):
        test_function.ground_truth_backend = self.ground_truth_backend
        test_function._is_ivy_backend_test = True
        test_function.test_data = self.test_data
        return test_function
