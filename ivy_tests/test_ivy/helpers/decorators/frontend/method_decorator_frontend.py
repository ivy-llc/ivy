# from hypothesis import strategies as st
from ivy_tests.test_ivy.helpers.test_parameter_flags import (
    BuiltAsVariableStrategy,
    BuiltNativeArrayStrategy,
)
from ivy_tests.test_ivy.helpers.test_parameter_flags import frontend_method_flags
from ivy_tests.test_ivy.helpers.decorators.base.method_decorator_base import (
    MethodHandlerBase,
)


class FrontendMethodHandler(MethodHandlerBase):
    def __init__(
        self,
        class_tree: str,
        init_tree: str,
        method_name: str,
        init_num_positional_args=None,
        init_native_arrays=BuiltNativeArrayStrategy,
        init_as_variable_flags=BuiltAsVariableStrategy,
        method_num_positional_args=None,
        method_native_arrays=BuiltNativeArrayStrategy,
        method_as_variable_flags=BuiltAsVariableStrategy,
        **_given_kwargs,
    ):
        self.init_tree = init_tree
        self.class_tree = class_tree
        self.method_name = method_name
        self._build_init_flags(
            init_num_positional_args, init_native_arrays, init_as_variable_flags
        )
        self._build_method_flags(
            method_num_positional_args,
            method_native_arrays,
            method_as_variable_flags,
        )
        self._given_kwargs = _given_kwargs

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

    @property
    def possible_args(self):
        return {
            "init_flags": self.init_flags,
            "method_flags": self.method_flags,
            # "frontend_method_data": st.just(frontend_helper_data),
        }
