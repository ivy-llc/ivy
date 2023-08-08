from hypothesis import strategies as st
from ivy_tests.test_ivy.helpers.test_parameter_flags import (
    BuiltAsVariableStrategy,
    BuiltNativeArrayStrategy,
)
from ivy_tests.test_ivy.helpers.structs import MethodData, SupportedDevicesDtypes
from ivy_tests.test_ivy.helpers.decorators.parameter_info_builder import (
    ParameterInfoStrategyBuilder,
)
from ivy_tests.test_ivy.helpers.test_parameter_flags import frontend_method_flags
from ivy_tests.test_ivy.helpers.decorators.base.method_handler_base import (
    MethodHandlerBase,
)


class FrontendMethodHandler(MethodHandlerBase):
    IVY_PREFIX = "ivy.functional.frontends"

    def __init__(
        self,
        class_tree: str,
        method_name: str,
        init_tree: str,
        init_num_positional_args=None,
        init_native_arrays=BuiltNativeArrayStrategy,
        init_as_variable_flags=BuiltAsVariableStrategy,
        method_num_positional_args=None,
        method_native_arrays=BuiltNativeArrayStrategy,
        method_as_variable_flags=BuiltAsVariableStrategy,
        **_given_kwargs,
    ):
        self._init_tree = init_tree
        self.class_tree = class_tree
        self.method_name = method_name

        if init_num_positional_args is None:
            init_strategy_builder = ParameterInfoStrategyBuilder.from_function(
                self.init_tree
            )
            init_num_positional_args = init_strategy_builder.build()

        self.init_flags = self._build_init_flags(
            init_num_positional_args, init_native_arrays, init_as_variable_flags
        )

        if method_num_positional_args is None:
            method_strategy_builder = ParameterInfoStrategyBuilder.from_method(
                self.method_tree
            )
            method_num_positional_args = method_strategy_builder.build()

        self.method_flags = self._build_method_flags(
            method_num_positional_args,
            method_native_arrays,
            method_as_variable_flags,
        )

        self._given_kwargs = _given_kwargs
        self.extra_test_data = self._build_extra_test_data()

    @property
    def init_tree(self):
        return self._append_ivy_prefix_to_tree(self._init_tree)

    @property
    def method_tree(self):
        return f"{self.class_tree}.{self.method_name}"

    @property
    def given_kwargs(self):
        return self._given_kwargs

    @property
    def possible_arguments(self):
        return {
            "init_flags": self.init_flags,
            "method_flags": self.method_flags,
            "frontend_method_data": st.just(self.extra_test_data),
        }

    @staticmethod
    def _build_init_flags(
        init_num_positional_args, init_as_variable_flags, init_native_arrays
    ):
        return frontend_method_flags(
            num_positional_args=init_num_positional_args,
            as_variable=init_as_variable_flags,
            native_arrays=init_native_arrays,
        )

    @staticmethod
    def _build_method_flags(
        method_num_positional_args, method_as_variable_flags, method_native_arrays
    ):
        return frontend_method_flags(
            num_positional_args=method_num_positional_args,
            as_variable=method_as_variable_flags,
            native_arrays=method_native_arrays,
        )

    def _build_extra_test_data(self):
        class_module_tree, _, class_name = self.class_tree.rpartition(".")
        init_tree, _, init_name = self._init_tree.rpartition(".")
        return MethodData(
            class_module_tree=class_module_tree,
            class_name=class_name,
            method_name=self.method_name,
            init_module_tree=init_tree,
            init_name=init_name,
        )

    @property
    def test_data(self):
        return SupportedDevicesDtypes(self.supported_devices_dtypes)

    def _add_test_attributes_to_test_function(self, fn):
        fn.test_data = self.test_data
        fn._is_ivy_method_test = True
        return fn
