# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.tensorflow as ivy_tf
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# binary_accuracy
@handle_cmd_line_args
@given(
    dtype_and_y_true_y_pred=helpers.dtype_and_values(
        available_dtypes=ivy_tf.valid_int_dtypes,
        min_num_dims=1,
        max_num_dims=5,
        min_value=0,
        max_value=1,
        shared_dtype=True,
        num_arrays=2,
    ),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.binary_accuracy"
    ),
    native_array=st.booleans(),
)
def test_tensorflow_binary_accuracy(
    dtype_and_y_true_y_pred, as_variable, num_positional_args, native_array, fw
):
    input_dtype, y_true_y_pred = dtype_and_y_true_y_pred
    y_true, y_pred = y_true_y_pred
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="keras.metrics.binary_accuracy",
        y_true=np.asarray(y_true, dtype=input_dtype[0]),
        y_pred=np.asarray(y_pred, dtype=input_dtype[1]),
        threshold=0.5,
    )
