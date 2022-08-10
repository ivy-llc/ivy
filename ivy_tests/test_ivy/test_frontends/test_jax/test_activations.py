import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.jax as ivy_jax


@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_jax.valid_numeric_dtypes),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.nn.relu"
    ),
    native_array=st.booleans(),
)
def test_jax_nn_relu(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_name="nn.relu",
        x=np.asarray(x, dtype=input_dtype),
    )


@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_jax.valid_float_dtypes),
    alpha=st.sampled_from(ivy_jax.valid_numeric_dtypes),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.nn.leaky_relu"
    ),
    native_array=st.booleans(),
)
def test_jax_nn_leaky_relu(
    dtype_and_x,
    alpha,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_name="nn.leaky_relu",
        x=np.asarray(x, dtype=input_dtype),
    )
