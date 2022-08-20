import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.tensorflow as ivy_tf


@given(
    dtype_array=helpers.dtype_and_values(available_dtypes=ivy_tf.valid_dtypes),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.exponential"
    ),
    native_array=st.booleans(),
)
def test_tensorflow_exponential(
    *, dtype_array, as_variable, num_positional_args, native_array, fw
):

    dtype, x = dtype_array

    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="keras.activations.exponential",
        x=np.asarray(x, dtype=dtype),
    )
