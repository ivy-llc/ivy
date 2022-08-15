import numpy as np
from hypothesis import given, strategies as st


# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.jax as ivy_jax


@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_jax.valid_numeric_dtypes, min_num_dims=2, min_dim_size=3
    ),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.nn.softmax"
    ),
    native_array=st.booleans(),
)
def test_jax_nn_softmax(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x
    data = np.asarray(x, dtype=input_dtype)
    # print("shape:", x.shape, ", x:", x)
    print("shape:", data.shape, ", data:", data)

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="nn.softmax",
        x=data,
        axis=data.ndim - 1,
    )
