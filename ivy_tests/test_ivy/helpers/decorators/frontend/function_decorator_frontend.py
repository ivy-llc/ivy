from ivy_tests.test_ivy.helpers.decorators.function_decorator_base import (
    FunctionHandler,
)

from typing import List
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

        self._build_test_flags(
            number_positional_args=number_positional_args,
            test_with_out=test_with_out,
            test_inplace=test_inplace,
            as_variable_flags=as_variable_flags,
            native_array_flags=native_array_flags,
            generate_frontend_arrays=generate_frontend_arrays,
        )

    def _init_aliases(self, aliases):
        if aliases is None:
            return None

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
