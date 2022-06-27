# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers

valid_float_dtypes = (ivy.bfloat16, ivy.float16, ivy.float32, ivy.float64)


# tan
@given(
    dtype_and_x=helpers.dtype_and_values(valid_float_dtypes),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.tan"),
    native_array=st.booleans(),
)
def test_jax_tan(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x

    helpers.test_frontend_function(
        input_dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        fw,
        "jax",
        "lax.tan",
        x=np.asarray(x, dtype=input_dtype),
    )
