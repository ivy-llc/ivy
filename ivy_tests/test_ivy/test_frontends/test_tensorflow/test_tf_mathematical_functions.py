# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.tensorflow as ivy_tf


# tan
@given(
    dtype_and_x=helpers.dtype_and_values(ivy_tf.valid_float_dtypes),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.tan"),
    native_array=st.booleans(),
)
def test_tensorflow_tan(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    fw
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        fw,
        "tensorflow",
        "tan",
        x=np.asarray(x, dtype=input_dtype)
    )
