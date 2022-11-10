"""Collection of tests for unified neural network activations."""

# global
from hypothesis import given
from hypothesis import strategies as st, assume

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args
import ivy

# GELU
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric")
    ),
    approximate=st.booleans(),
    num_positional_args_init=helpers.num_positional_args(fn_name="GELU.__init__"),
    num_positional_args_method=helpers.num_positional_args(fn_name="GELU._forward"),
)
def test_gelu(
    *,
    dtype_and_x,
    approximate,
    num_positional_args_init,
    num_positional_args_method,
    as_variable,
    native_array,
    container,
    fw
):
    input_dtype, x = dtype_and_x
    assume(not (fw == "torch" and "float16" in input_dtype))
    helpers.test_method(
        input_dtypes_init=input_dtype,
        num_positional_args_init=num_positional_args_init,
        all_as_kwargs_np_init={"approximate": approximate},
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=num_positional_args_method,
        native_array_flags_method=native_array,
        container_flags_method=container,
        all_as_kwargs_np_method={"x": x[0]},
        class_name="GELU",
    )


# GEGLU
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        large_abs_safety_factor=4,
        small_abs_safety_factor=4,
        safety_factor_scale="log",
    ),
    num_positional_args_init=helpers.num_positional_args(fn_name="GEGLU.__init__"),
    num_positional_args_method=helpers.num_positional_args(fn_name="GEGLU._forward"),
)
def test_geglu(
    *,
    dtype_and_x,
    num_positional_args_init,
    num_positional_args_method,
    as_variable,
    native_array,
    container,
):
    input_dtype, x = dtype_and_x
    # float16 is somehow generated
    assume("float16" not in input_dtype)
    # last dim must be even, this could replaced with a private helper
    assume(x[0].shape[-1] % 2 == 0)
    helpers.test_method(
        input_dtypes_init=input_dtype,
        num_positional_args_init=num_positional_args_init,
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=num_positional_args_method,
        native_array_flags_method=native_array,
        container_flags_method=container,
        all_as_kwargs_np_method={"inputs": x[0]},
        class_name="GEGLU",
        atol_=1e-3,
    )
