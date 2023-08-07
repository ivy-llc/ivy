from ivy_tests.test_ivy.helpers.test_parameter_flags import (
    BuiltAsVariableStrategy,
    BuiltNativeArrayStrategy,
    BuiltGradientStrategy,
    BuiltContainerStrategy,
    BuiltCompileStrategy,
)


class HandleBackendMethod:
    def __init__(
        self,
        init_tree: str,
        method_tree: str,
        ground_truth_backend: str = "tensorflow",
        test_gradients=BuiltGradientStrategy,
        test_compile=BuiltCompileStrategy,
        init_num_positional_args=None,
        init_native_arrays=BuiltNativeArrayStrategy,
        init_as_variable_flags=BuiltAsVariableStrategy,
        method_num_positional_args=None,
        method_native_arrays=BuiltNativeArrayStrategy,
        method_as_variable_flags=BuiltAsVariableStrategy,
        method_container_flags=BuiltContainerStrategy,
        **_given_kwargs,
    ):
        pass
