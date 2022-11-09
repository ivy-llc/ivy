"""Collection of tests for unified neural network activations."""

# global
from hypothesis import strategies as st, assume

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_method
from ivy_tests.test_ivy.helpers.function_testing import (
    ContainerFlags,
    NativeArrayFlags,
    AsVariableFlags,
    NumPositionalArg,
)


# GELU
@handle_method(
    method_tree="stateful.activations.GELU.__call__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric")
    ),
    approximate=st.booleans(),
    method_num_positional_args=helpers.num_positional_args(fn_name="GELU._forward"),
)
def test_gelu(
    *,
    dtype_and_x,
    approximate,
    init_num_positional_args: NumPositionalArg,
    method_num_positional_args,
    init_as_variable: AsVariableFlags,
    init_native_array: NativeArrayFlags,
    method_as_variable: AsVariableFlags,
    method_native_array: NativeArrayFlags,
    method_container: ContainerFlags,
    method_name,
    class_name,
):
    input_dtype, x = dtype_and_x
    helpers.test_method(
        input_dtypes_init=input_dtype,
        all_as_kwargs_np_init={"approximate": approximate},
        input_dtypes_method=input_dtype,
        as_variable_flags_init=init_as_variable,
        num_positional_args_init=init_num_positional_args,
        native_array_flags_init=init_native_array,
        as_variable_flags_method=method_as_variable,
        num_positional_args_method=method_num_positional_args,
        native_array_flags_method=method_native_array,
        container_flags_method=method_container,
        all_as_kwargs_np_method={"x": x[0]},
        class_name=class_name,
        method_name=method_name,
    )


# GEGLU
@handle_method(
    method_tree="stateful.activations.GEGLU.__call__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        large_abs_safety_factor=4,
        small_abs_safety_factor=4,
        safety_factor_scale="log",
    ),
    num_positional_args_method=helpers.num_positional_args(fn_name="GEGLU._forward"),
)
def test_geglu(
    *,
    dtype_and_x,
    num_positional_args_init: NumPositionalArg,
    num_positional_args_method,
    method_as_variable_flags: AsVariableFlags,
    method_native_array_flags: NativeArrayFlags,
    method_container_flags: ContainerFlags,
    class_name,
    method_name,
):
    input_dtype, x = dtype_and_x
    # last dim must be even, this could replaced with a private helper
    assume(x[0].shape[-1] % 2 == 0)
    helpers.test_method(
        input_dtypes_init=input_dtype,
        num_positional_args_init=num_positional_args_init,
        input_dtypes_method=input_dtype,
        as_variable_flags_method=method_as_variable_flags,
        num_positional_args_method=num_positional_args_method,
        native_array_flags_method=method_native_array_flags,
        container_flags_method=method_container_flags,
        all_as_kwargs_np_method={"inputs": x[0]},
        class_name=class_name,
        method_name=method_name,
        atol_=1e-3,
    )
