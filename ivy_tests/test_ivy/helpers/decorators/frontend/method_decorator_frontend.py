from hypothesis import strategies as st
from ivy_tests.test_ivy.helpers.test_parameter_flags import (
    BuiltAsVariableStrategy,
    BuiltNativeArrayStrategy,
)
from ivy_tests.test_ivy.helpers.structs import MethodData
from ivy_tests.test_ivy.helpers.test_parameter_flags import frontend_method_flags
from ivy_tests.test_ivy.helpers.decorators.base.method_decorator_base import (
    MethodHandlerBase,
)


class FrontendMethodHandler(MethodHandlerBase):
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
            init_num_positional_args = self._build_init_num_positional_args_strategy()

        self._build_init_flags(
            init_num_positional_args, init_native_arrays, init_as_variable_flags
        )

        if method_num_positional_args is None:
            method_num_positional_args = (
                self._build_method_num_positional_args_strategy()
            )

        self._build_method_flags(
            method_num_positional_args,
            method_native_arrays,
            method_as_variable_flags,
        )
        self._given_kwargs = _given_kwargs
        self._build_test_data()

    @property
    def init_tree(self):
        return self._append_ivy_to_fn_tree(self._init_tree)

    @property
    def method_tree(self):
        return f"{self.class_tree}.{self.method_name}"

    def _build_init_num_positional_args_strategy(self):
        return self._build_parameters_info_dict_from_function(self.init_tree)

    def _build_method_num_positional_args_strategy(self):
        method_tree = f"{self.class_tree}.{self.method_name}"
        return self._build_parameters_info_dict_from_method(method_tree)

    def _append_ivy_to_fn_tree(self, fn_tree):
        return "ivy.functional.frontends." + fn_tree

    def _build_init_flags(
        self, init_num_positional_args, init_as_variable_flags, init_native_arrays
    ):
        self.init_flags = frontend_method_flags(
            num_positional_args=init_num_positional_args,
            as_variable=init_as_variable_flags,
            native_arrays=init_native_arrays,
        )

    def _build_method_flags(
        self, method_num_positional_args, method_as_variable_flags, method_native_arrays
    ):
        self.method_flags = frontend_method_flags(
            num_positional_args=method_num_positional_args,
            as_variable=method_as_variable_flags,
            native_arrays=method_native_arrays,
        )

    def _add_test_attributes_to_test_function(self, fn):
        fn.test_data = self.test_data
        fn._is_ivy_method_test = True
        return fn

    def _build_test_data(self):
        class_module_tree, _, class_name = self.class_tree.rpartition(".")
        init_tree, _, init_name = self.init_tree.rpartition(".")
        supported_device_dtypes = self._build_supported_devices_dtypes()
        self.test_data = MethodData(
            class_module_tree=class_module_tree,
            class_name=class_name,
            method_name=self.method_name,
            init_module_tree=init_tree,
            init_name=init_name,
            method_supported_device_dtypes=supported_device_dtypes,
        )

    @property
    def possible_args(self):
        return {
            "init_flags": self.init_flags,
            "method_flags": self.method_flags,
            "frontend_method_data": st.just(self.test_data),
        }
