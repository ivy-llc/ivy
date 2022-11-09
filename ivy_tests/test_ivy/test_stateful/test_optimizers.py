# For Review
"""Collection of tests for Ivy optimizers."""

# global
from hypothesis import given
from hypothesis import strategies as st

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
    lr=helpers.floats(min_value=0.0, max_value=1.0),
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
    device,
):
    input_dtype, x = dtype_and_x
    helpers.test_method(
        init_num_positional_args=num_positional_args_init,
        init_all_as_kwargs_np={
            "lr": lr,
            "inplace": inplace,
            "stop_gradients": stop_gradients,
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=num_positional_args_method,
        method_native_array_flags=native_array,
        method_container_flags=container,
        method_all_as_kwargs_np={
            "v": x[0],
            "grads": x[1],
        },
        class_name="SGD",
        method_name="step",
        atol_=1e-4,
        device_=device,
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
    lr=helpers.floats(min_value=0.0, max_value=1.0),
    decay_lambda=helpers.floats(min_value=0.0, max_value=1.0),
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
    device,
):
    input_dtype, x = dtype_and_x
    helpers.test_method(
        init_num_positional_args=num_positional_args_init,
        init_all_as_kwargs_np={
            "lr": lr,
            "decay_lambda": decay_lambda,
            "inplace": inplace,
            "stop_gradients": stop_gradients,
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=num_positional_args_method,
        method_native_array_flags=native_array,
        method_container_flags=container,
        method_all_as_kwargs_np={
            "v": x[0],
            "grads": x[1],
        },
        class_name="LARS",
        method_name="step",
        device_=device,
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
    lr=helpers.floats(min_value=0.1e-6, max_value=1.0),
    beta1=helpers.floats(min_value=0.1e-6, max_value=1.0),
    beta2=helpers.floats(min_value=0.1e-6, max_value=1.0),
    epsilon=helpers.floats(min_value=1e-07, max_value=1.0),
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
):
    input_dtype, x = dtype_and_x
    helpers.test_method(
        init_num_positional_args=num_positional_args_init,
        init_all_as_kwargs_np={
            "lr": lr,
            "beta1": beta1,
            "beta2": beta2,
            "epsilon": epsilon,
            "inplace": inplace,
            "stop_gradients": stop_gradients,
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=num_positional_args_method,
        method_native_array_flags=native_array,
        method_container_flags=container,
        method_all_as_kwargs_np={
            "v": x[0],
            "grads": x[1],
        },
        class_name="Adam",
        method_name="step",
        device_=device,
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
    lr=helpers.floats(min_value=0.1e-6, max_value=1.0),
    beta1=helpers.floats(min_value=0.1e-6, max_value=1.0),
    beta2=helpers.floats(min_value=0.1e-6, max_value=1.0),
    epsilon=helpers.floats(min_value=1e-07, max_value=1.0),
    max_trust_ratio=helpers.floats(min_value=0.0, max_value=10),
    decay_lambda=helpers.floats(min_value=0.0, max_value=1.0),
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
):
    input_dtype, x = dtype_and_x
    helpers.test_method(
        init_num_positional_args=num_positional_args_init,
        init_all_as_kwargs_np={
            "lr": lr,
            "beta1": beta1,
            "beta2": beta2,
            "epsilon": epsilon,
            "max_trust_ratio": max_trust_ratio,
            "decay_lambda": decay_lambda,
            "inplace": inplace,
            "stop_gradients": stop_gradients,
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=num_positional_args_method,
        method_native_array_flags=native_array,
        method_container_flags=container,
        method_all_as_kwargs_np={
            "v": x[0],
            "grads": x[1],
        },
        class_name="LAMB",
        method_name="step",
        device_=device,
    )
