from hypothesis import strategies as st
from ivy_tests.test_ivy.helpers.decorators.function_decorator_base import (
    FunctionHandler,
)

from ivy_tests.test_ivy.helpers.structs import FunctionData
from typing import List, Callable, Any
from ivy_tests.test_ivy.helpers.test_parameter_flags import (
    build_frontend_function_flags,
)
from ivy_tests.test_ivy.helpers.test_parameter_flags import (
    BuiltAsVariableStrategy,
    BuiltNativeArrayStrategy,
    BuiltWithOutStrategy,
    BuiltInplaceStrategy,
    BuiltFrontendArrayStrategy,
)


class FrontendFunctionHandler(FunctionHandler):
    def __init__(
        self,
        fn_tree: str,
        aliases: List[str] = None,
        number_positional_args=None,
        test_with_out=BuiltWithOutStrategy,
        test_inplace=BuiltInplaceStrategy,
        as_variable_flags=BuiltAsVariableStrategy,
        native_array_flags=BuiltNativeArrayStrategy,
        generate_frontend_arrays=BuiltFrontendArrayStrategy,
        **_given_kwargs,
    ):
        self.fn_tree = self._append_ivy_to_fn_tree(fn_tree)
        self.aliases = self._init_aliases(aliases)
        self._given_kwargs = _given_kwargs

        if number_positional_args is None:
            number_positional_args = self._build_num_positional_arguments_strategy()

        self._build_test_flags(
            number_positional_args=number_positional_args,
            test_with_out=test_with_out,
            test_inplace=test_inplace,
            as_variable_flags=as_variable_flags,
            native_array_flags=native_array_flags,
            generate_frontend_arrays=generate_frontend_arrays,
        )
        self._build_test_data()

    def _build_test_data(self):
        module_tree, fn_name = self._partition_fn_tree(self.fn_tree)
        supported_device_dtypes = self._get_supported_devices_dtypes(self.fn_tree)
        self.test_data = FunctionData(
            module_tree=module_tree,
            fn_name=fn_name,
            supported_device_dtypes=supported_device_dtypes,
        )

    def _build_fn_tree_strategy(self):
        if self.aliases is None:
            return st.just(self.fn_tree)
        else:
            return st.sampled_from([self.fn_tree] + self.aliases)

    @property
    def possible_args(self):
        return {
            "fn_tree": self._build_fn_tree_strategy(),
            "test_flags": self.test_flags,
        }

    def _init_aliases(self, aliases):
        if aliases is None:
            return None

        # TODO use enumeurate?
        for i in range(len(aliases)):
            aliases[i] = self._append_ivy_to_fn_tree(aliases[i])

    def _append_ivy_to_fn_tree(self, fn_tree):
        return "ivy.functional.frontends." + fn_tree

    def _build_test_flags(
        self,
        number_positional_args,
        test_with_out,
        test_inplace,
        as_variable_flags,
        native_array_flags,
        generate_frontend_arrays,
    ):
        self.test_flags = build_frontend_function_flags(
            num_positional_args=number_positional_args,
            with_out=test_with_out,
            inplace=test_inplace,
            as_variable=as_variable_flags,
            native_arrays=native_array_flags,
            generate_frontend_arrays=generate_frontend_arrays,
        )

    def _add_test_attributes_to_test_function(self, fn: Callable[..., Any]):
        fn._is_ivy_frontend_test = True
        fn.test_data = self.test_data
        return fn
