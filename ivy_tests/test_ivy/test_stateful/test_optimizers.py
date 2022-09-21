# For Review
"""Collection of tests for Ivy optimizers."""

# global
from hypothesis import given
from hypothesis import strategies as st
import numpy as np

# local
import ivy.functional.backends.numpy as ivy_np
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# sgd
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        allow_inf=False,
        shared_dtype=True,
    ),
    lr=st.floats(min_value=0.0, max_value=1.0),
    inplace=st.booleans(),
    stop_gradients=st.booleans(),
    num_positional_args_init=helpers.num_positional_args(fn_name="SGD.__init__"),
    num_positional_args_method=helpers.num_positional_args(fn_name="SGD._step"),
)
def test_sgd_optimizer(
    dtype_and_x,
    lr,
    inplace,
    stop_gradients,
    num_positional_args_init,
    num_positional_args_method,
    as_variable,
    native_array,
    container,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_method(
        num_positional_args_init=num_positional_args_init,
        num_positional_args_method=num_positional_args_method,
        all_as_kwargs_np_init={
            "lr": lr,
            "inplace": inplace,
            "stop_gradients": stop_gradients,
        },
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        native_array_flags_method=native_array,
        container_flags_method=container,
        all_as_kwargs_np_method={
            "v": np.asarray(x[0], dtype=input_dtype[0]),
            "grads": np.asarray(x[1], dtype=input_dtype[1]),
        },
        fw=fw,
        class_name="SGD",
        method_name="step",
    )


# lars
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        allow_inf=False,
        shared_dtype=True,
    ),
    inplace=st.booleans(),
    lr=st.floats(min_value=0.0, max_value=1.0),
    decay_lambda=st.floats(min_value=0.0, max_value=1.0),
    stop_gradients=st.booleans(),
    num_positional_args_init=helpers.num_positional_args(fn_name="LARS.__init__"),
    num_positional_args_method=helpers.num_positional_args(fn_name="LARS._step"),
)
def test_lars_optimizer(
    dtype_and_x,
    lr,
    decay_lambda,
    inplace,
    stop_gradients,
    num_positional_args_init,
    num_positional_args_method,
    as_variable,
    native_array,
    container,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_method(
        num_positional_args_init=num_positional_args_init,
        num_positional_args_method=num_positional_args_method,
        all_as_kwargs_np_init={
            "lr": lr,
            "decay_lambda": decay_lambda,
            "inplace": inplace,
            "stop_gradients": stop_gradients,
        },
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        native_array_flags_method=native_array,
        container_flags_method=container,
        all_as_kwargs_np_method={
            "v": np.asarray(x[0], dtype=input_dtype[0]),
            "grads": np.asarray(x[1], dtype=input_dtype[1]),
        },
        fw=fw,
        class_name="LARS",
        method_name="step",
    )


# adam
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_float_dtypes[1:],
        num_arrays=2,
        allow_inf=False,
        shared_dtype=True,
    ),
    lr=st.floats(min_value=0.0, max_value=1.0),
    beta1=st.floats(min_value=0.0, max_value=1.0),
    beta2=st.floats(min_value=0.0, max_value=1.0),
    epsilon=st.floats(min_value=1e-07, max_value=1.0),
    inplace=st.booleans(),
    stop_gradients=st.booleans(),
    num_positional_args_init=helpers.num_positional_args(fn_name="Adam.__init__"),
    num_positional_args_method=helpers.num_positional_args(fn_name="Adam._step"),
)
def test_adam_optimizer(
    dtype_and_x,
    lr,
    beta1,
    beta2,
    epsilon,
    inplace,
    stop_gradients,
    device,
    num_positional_args_init,
    num_positional_args_method,
    as_variable,
    native_array,
    container,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_method(
        num_positional_args_init=num_positional_args_init,
        num_positional_args_method=num_positional_args_method,
        all_as_kwargs_np_init={
            "lr": lr,
            "beta1": beta1,
            "beta2": beta2,
            "epsilon": epsilon,
            "inplace": inplace,
            "stop_gradients": stop_gradients,
        },
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        native_array_flags_method=native_array,
        container_flags_method=container,
        device_=device,
        all_as_kwargs_np_method={
            "v": np.asarray(x[0], dtype=input_dtype[0]),
            "grads": np.asarray(x[1], dtype=input_dtype[1]),
        },
        fw=fw,
        class_name="Adam",
        method_name="step",
    )


# lamb
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_float_dtypes[1:],
        num_arrays=2,
        allow_inf=False,
        shared_dtype=True,
    ),
    lr=st.floats(min_value=0.0, max_value=1.0),
    beta1=st.floats(min_value=0.0, max_value=1.0),
    beta2=st.floats(min_value=0.0, max_value=1.0),
    epsilon=st.floats(min_value=1e-07, max_value=1.0),
    max_trust_ratio=st.floats(min_value=0.0, max_value=10),
    decay_lambda=st.floats(min_value=0.0, max_value=1.0),
    inplace=st.booleans(),
    stop_gradients=st.booleans(),
    num_positional_args_init=helpers.num_positional_args(fn_name="LAMB.__init__"),
    num_positional_args_method=helpers.num_positional_args(fn_name="LAMB._step"),
)
def test_lamb_optimizer(
    dtype_and_x,
    lr,
    beta1,
    beta2,
    epsilon,
    max_trust_ratio,
    decay_lambda,
    inplace,
    stop_gradients,
    device,
    num_positional_args_init,
    num_positional_args_method,
    as_variable,
    native_array,
    container,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_method(
        num_positional_args_init=num_positional_args_init,
        num_positional_args_method=num_positional_args_method,
        all_as_kwargs_np_init={
            "lr": lr,
            "beta1": beta1,
            "beta2": beta2,
            "epsilon": epsilon,
            "max_trust_ratio": max_trust_ratio,
            "decay_lambda": decay_lambda,
            "inplace": inplace,
            "stop_gradients": stop_gradients,
        },
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        native_array_flags_method=native_array,
        container_flags_method=container,
        device_=device,
        all_as_kwargs_np_method={
            "v": np.asarray(x[0], dtype=input_dtype[0]),
            "grads": np.asarray(x[1], dtype=input_dtype[1]),
        },
        fw=fw,
        class_name="LAMB",
        method_name="step",
    )
