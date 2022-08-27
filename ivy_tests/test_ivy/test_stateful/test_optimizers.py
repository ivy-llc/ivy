"""Collection of tests for Ivy optimizers."""

# global
# from matplotlib.style import available
# import pytest
from hypothesis import given
from hypothesis import strategies as st
import numpy as np

# local
# import ivy
# from ivy.container import Container
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# sgd
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes,
        num_arrays=2,
        shared_dtype=True,
    ),
    with_v=st.booleans(),
    lr=st.floats(min_value=0.0, max_value=1.0),
    inplace=st.booleans(),
    stop_gradients=st.booleans(),
    compile_on_next_step=st.booleans(),
    num_positional_args_init=helpers.num_positional_args(fn_name="SGD.__init__"),
    num_positional_args_method=helpers.num_positional_args(fn_name="SGD._step"),
)
def test_sgd_optimizer(
    dtype_and_x,
    with_v,
    lr,
    inplace,
    stop_gradients,
    compile_on_next_step,
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
            "compile_on_next_step": compile_on_next_step,
        },
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        native_array_flags_method=native_array,
        container_flags_method=container,
        method_with_v=with_v,
        all_as_kwargs_np_method={
            "v": np.asarray(x[0], dtype=input_dtype[0]),
            "grads": np.asarray(x[1], dtype=input_dtype[1]),
        },
        fw=fw,
        class_name="SGD",
        method_name="_step",
    )


# lars
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_float_dtypes,
        num_arrays=2,
        shared_dtype=True,
    ),
    with_v=st.booleans(),
    inplace=st.booleans(),
    lr=st.floats(min_value=0.0, max_value=1.0),
    decay_lamda=st.floats(min_value=0.0, max_value=1.0),
    stop_gradients=st.booleans(),
    compile_on_next_step=st.booleans(),
    num_positional_args_init=helpers.num_positional_args(fn_name="LARS.__init__"),
    num_positional_args_method=helpers.num_positional_args(fn_name="LARS._step"),
)
def test_lars_optimizer(
    dtype_and_x,
    with_v,
    lr,
    decay_lamda,
    inplace,
    stop_gradients,
    compile_on_next_step,
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
            # "decay_lamda": decay_lamda,
            "inplace": inplace,
            "stop_gradients": stop_gradients,
            "compile_on_next_step": compile_on_next_step,
        },
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        native_array_flags_method=native_array,
        container_flags_method=container,
        method_with_v=with_v,
        all_as_kwargs_np_method={
            "v": np.asarray(x[0], dtype=input_dtype[0]),
            "grads": np.asarray(x[1], dtype=input_dtype[1]),
        },
        fw=fw,
        class_name="LARS",
        method_name="_step",
    )


# adam
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes,
        num_arrays=2,
        shared_dtype=True,
    ),
    lr=st.floats(min_value=0.0, max_value=1.0),
    inplace=st.booleans(),
    stop_gradients=st.booleans(),
    compile_on_next_step=st.booleans(),
    device=st.sampled_from(["cpu", "cuda"]),
    with_v=st.booleans(),
    num_positional_args_init=helpers.num_positional_args(fn_name="Adam.__init__"),
    num_positional_args_method=helpers.num_positional_args(fn_name="Adam._step"),
)
def test_adam_optimizer(
    dtype_and_x,
    lr,
    inplace,
    stop_gradients,
    compile_on_next_step,
    device,
    with_v,
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
            "compile_on_next_step": compile_on_next_step,
        },
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        native_array_flags_method=native_array,
        container_flags_method=container,
        method_with_v=with_v,
        device_=device,
        all_as_kwargs_np_method={
            "v": np.asarray(x[0], dtype=input_dtype[0]),
            "grads": np.asarray(x[1], dtype=input_dtype[1]),
        },
        fw=fw,
        class_name="Adam",
        method_name="_step",
    )


# lamb
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_float_dtypes,
        num_arrays=2,
        shared_dtype=True,
    ),
    lr=st.floats(min_value=0.0, max_value=1.0),
    inplace=st.booleans(),
    stop_gradients=st.booleans(),
    compile_on_next_step=st.booleans(),
    device=st.sampled_from(["cpu", "cuda"]),
    with_v=st.booleans(),
    num_positional_args_init=helpers.num_positional_args(fn_name="LAMB.__init__"),
    num_positional_args_method=helpers.num_positional_args(fn_name="LAMB._step"),
)
def test_lamb_optimizer(
    dtype_and_x,
    lr,
    inplace,
    stop_gradients,
    compile_on_next_step,
    device,
    with_v,
    num_positional_args_init,
    num_positional_args_method,
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
            "compile_on_next_step": compile_on_next_step,
        },
        input_dtypes_method=input_dtype,
        as_variable_flags_method=with_v,
        native_array_flags_method=False,
        container_flags_method=False,
        method_with_v=with_v,
        device_=device,
        all_as_kwargs_np_method={
            "v": np.asarray(x[0], dtype=input_dtype[0]),
            "grads": np.asarray(x[1], dtype=input_dtype[1]),
        },
        fw=fw,
        class_name="LAMB",
        method_name="_step",
    )
