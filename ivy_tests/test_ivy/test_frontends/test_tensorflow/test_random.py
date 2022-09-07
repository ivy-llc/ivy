# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers

# import ivy.functional.backends.numpy as ivy_np
import ivy.functional.backends.tensorflow as ivy_tf
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_tf.valid_float_dtypes,
        allow_inf=False,
        min_num_dims=1,
        min_dim_size=2,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.shuffle"
    ),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_tensorflow_shuffle(
    dtype_and_x, seed, as_variable, num_positional_args, native_array, fw
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        test_values=False,
        fw=fw,
        frontend="tensorflow",
        fn_tree="random.shuffle",
        value=np.asarray(x, dtype=input_dtype),
        seed=seed,
    )


@handle_cmd_line_args
@given(
    dtype_and_low=helpers.dtype_and_values(
        available_dtypes=ivy_tf.valid_float_dtypes,
        min_value=-1000,
        max_value=100,
        allow_inf=False,
    ),
    dtype_and_high=helpers.dtype_and_values(
        available_dtypes=ivy_tf.valid_float_dtypes,
        min_value=101,
        max_value=1000,
        allow_inf=False,
    ),
    shape=helpers.get_shape(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.uniform"
    ),
)
def test_tensorflow_uniform(
    dtype_and_low, dtype_and_high, shape, as_variable, num_positional_args, native_array, fw
):
    low_dtype, low = dtype_and_low
    high_dtype, high = dtype_and_high
    # If low and high are numerics and not arrays, shape=shape else shape=None
    if isinstance(low, (int, float)) and isinstance(high, (int, float)):
        shape = shape
    else:
        shape = None
    helpers.test_frontend_function(
        input_dtypes=[low_dtype, high_dtype],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="random.uniform",
        shape=shape,
        minval=np.asarray(low, dtype=low_dtype),
        maxval=np.asarray(high, dtype=high_dtype),
    )