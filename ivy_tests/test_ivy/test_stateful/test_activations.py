"""Collection of tests for unified neural network activations."""

# global
from hypothesis import strategies as st, assume

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_method
import ivy_tests.test_ivy.helpers.test_parameter_flags as pf


# GELU
@handle_method(
    method_tree="stateful.activations.GELU.__call__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
    approximate=st.booleans(),
    method_num_positional_args=helpers.num_positional_args(fn_name="GELU._forward"),
)
def test_gelu(
    *,
    dtype_and_x,
    approximate,
    init_num_positional_args: pf.NumPositionalArg,
    method_num_positional_args,
    init_as_variable: pf.AsVariableFlags,
    init_native_array: pf.NativeArrayFlags,
    method_as_variable: pf.AsVariableFlags,
    method_native_array: pf.NativeArrayFlags,
    method_container: pf.ContainerFlags,
    method_name,
    class_name,
    ground_truth_backend,
):
    input_dtype, x = dtype_and_x
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_input_dtypes=input_dtype,
        init_as_variable_flags=init_as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=init_native_array,
        init_all_as_kwargs_np={"approximate": approximate},
        method_input_dtypes=input_dtype,
        method_as_variable_flags=method_as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=method_native_array,
        method_container_flags=method_container,
        method_all_as_kwargs_np={"x": x[0]},
        class_name=class_name,
        method_name=method_name,
        atol_=1e-2,
        rtol_=1e-2,
    )


# GEGLU
@handle_method(
    method_tree="stateful.activations.GEGLU.__call__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
    num_positional_args_method=helpers.num_positional_args(fn_name="GEGLU._forward"),
)
def test_geglu(
    *,
    dtype_and_x,
    num_positional_args_init: pf.NumPositionalArg,
    num_positional_args_method,
    method_as_variable_flags: pf.AsVariableFlags,
    method_native_array_flags: pf.NativeArrayFlags,
    method_container_flags: pf.ContainerFlags,
    class_name,
    method_name,
    ground_truth_backend,
):
    input_dtype, x = dtype_and_x
    # last dim must be even, this could replaced with a private helper
    assume(x[0].shape[-1] % 2 == 0)
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_input_dtypes=input_dtype,
        init_num_positional_args=num_positional_args_init,
        method_input_dtypes=input_dtype,
        method_as_variable_flags=method_as_variable_flags,
        method_num_positional_args=num_positional_args_method,
        method_native_array_flags=method_native_array_flags,
        method_container_flags=method_container_flags,
        method_all_as_kwargs_np={"inputs": x[0]},
        class_name=class_name,
        method_name=method_name,
        rtol_=1e-2,
        atol_=1e-2,
    )
