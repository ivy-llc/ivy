# For Review
"""Collection of tests for Ivy optimizers."""

# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy_tests.test_ivy.helpers.test_parameter_flags as pf
from ivy_tests.test_ivy.helpers import handle_method
from ivy_tests.test_ivy.test_functional.test_core.test_gradients import (
    get_gradient_arguments_with_lr,
)


# sgd
@handle_method(
    method_tree="SGD._step",
    dtype_x_lr=get_gradient_arguments_with_lr(
        min_value=-1e5,
        max_value=1e5,
        num_arrays=2,
        float_lr=True,
    ),
    inplace=st.booleans(),
    stop_gradients=st.booleans(),
)
def test_sgd_optimizer(
    dtype_x_lr,
    inplace,
    stop_gradients,
    num_positional_args_init: pf.NumPositionalArg,
    num_positional_args_method: pf.NumPositionalArg,
    method_as_variable_flags: pf.AsVariableFlags,
    method_native_array_flags: pf.NativeArrayFlags,
    method_container_flags: pf.ContainerFlags,
    test_gradients: pf.BuiltGradientStrategy,
    on_device,
    class_name,
    method_name,
    ground_truth_backend,
):
    input_dtype, x, lr = dtype_x_lr
    xs_grad_idxs = [[0, 0]] if num_positional_args_method else [[1, "v"]]
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_num_positional_args=num_positional_args_init,
        init_all_as_kwargs_np={
            "lr": lr,
            "inplace": inplace,
            "stop_gradients": stop_gradients,
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=method_as_variable_flags,
        method_num_positional_args=num_positional_args_method,
        method_native_array_flags=method_native_array_flags,
        method_container_flags=method_container_flags,
        method_all_as_kwargs_np={
            "v": x[0],
            "grads": x[1],
        },
        class_name=class_name,
        method_name=method_name,
        rtol_=1e-2,
        atol_=1e-2,
        test_gradients=test_gradients,
        xs_grad_idxs=xs_grad_idxs,
        device_=on_device,
    )


# lars
@handle_method(
    method_tree="LARS._step",
    dtype_x_lr=get_gradient_arguments_with_lr(num_arrays=2, float_lr=True),
    inplace=st.booleans(),
    decay_lambda=helpers.floats(min_value=1e-2, max_value=1.0),
    stop_gradients=st.booleans(),
)
def test_lars_optimizer(
    dtype_x_lr,
    decay_lambda,
    inplace,
    stop_gradients,
    num_positional_args_init: pf.NumPositionalArg,
    num_positional_args_method: pf.NumPositionalArg,
    method_as_variable_flags: pf.AsVariableFlags,
    method_native_array_flags: pf.NativeArrayFlags,
    method_container_flags: pf.ContainerFlags,
    test_gradients: pf.BuiltGradientStrategy,
    on_device,
    class_name,
    method_name,
    ground_truth_backend,
):
    input_dtype, x, lr = dtype_x_lr
    if "bfloat16" in input_dtype:
        test_gradients = False
    xs_grad_idxs = [[0, 0]] if num_positional_args_method else [[1, "v"]]
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_num_positional_args=num_positional_args_init,
        init_all_as_kwargs_np={
            "lr": lr,
            "decay_lambda": decay_lambda,
            "inplace": inplace,
            "stop_gradients": stop_gradients,
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=method_as_variable_flags,
        method_num_positional_args=num_positional_args_method,
        method_native_array_flags=method_native_array_flags,
        method_container_flags=method_container_flags,
        method_all_as_kwargs_np={
            "v": x[0],
            "grads": x[1],
        },
        class_name=class_name,
        method_name=method_name,
        rtol_=1e-1,
        atol_=1e-1,
        test_gradients=test_gradients,
        xs_grad_idxs=xs_grad_idxs,
        device_=on_device,
    )


# adam
@handle_method(
    method_tree="Adam._step",
    dtype_x_lr=get_gradient_arguments_with_lr(
        min_value=1e-05,
        max_value=1e08,
        num_arrays=2,
        float_lr=True,
        large_abs_safety_factor=2,
        small_abs_safety_factor=2,
    ),
    beta1_n_beta2_n_epsilon=helpers.lists(
        arg=helpers.floats(min_value=1e-1, max_value=1),
        min_size=3,
        max_size=3,
    ),
    inplace=st.booleans(),
    stop_gradients=st.booleans(),
    num_positional_args_method=helpers.num_positional_args(fn_name="Adam._step"),
)
def test_adam_optimizer(
    dtype_x_lr,
    beta1_n_beta2_n_epsilon,
    inplace,
    stop_gradients,
    on_device,
    num_positional_args_init: pf.NumPositionalArg,
    num_positional_args_method: pf.NumPositionalArg,
    method_as_variable_flags: pf.AsVariableFlags,
    method_native_array_flags: pf.NativeArrayFlags,
    method_container_flags: pf.ContainerFlags,
    test_gradients: pf.BuiltGradientStrategy,
    class_name,
    method_name,
    ground_truth_backend,
):
    input_dtype, x, lr = dtype_x_lr
    beta1, beta2, epsilon = beta1_n_beta2_n_epsilon
    xs_grad_idxs = [[0, 0]] if num_positional_args_method else [[1, "v"]]
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
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
        method_as_variable_flags=method_as_variable_flags,
        method_num_positional_args=num_positional_args_method,
        method_native_array_flags=method_native_array_flags,
        method_container_flags=method_container_flags,
        method_all_as_kwargs_np={
            "v": x[0],
            "grads": x[1],
        },
        class_name=class_name,
        method_name=method_name,
        rtol_=1e-1,
        atol_=1e-1,
        test_gradients=test_gradients,
        xs_grad_idxs=xs_grad_idxs,
        device_=on_device,
    )


# lamb
@handle_method(
    method_tree="LAMB._step",
    dtype_x_lr=get_gradient_arguments_with_lr(
        min_value=-1e5,
        max_value=1e5,
        num_arrays=2,
        float_lr=True,
    ),
    beta1_n_beta2_n_epsilon_n_lambda=helpers.lists(
        arg=helpers.floats(
            min_value=1e-2,
            max_value=1.0,
        ),
        min_size=4,
        max_size=4,
    ),
    mtr=st.one_of(
        helpers.ints(min_value=1, max_value=10),
        st.floats(min_value=1e-2, max_value=10, exclude_min=True),
    ),
    inplace=st.booleans(),
    stop_gradients=st.booleans(),
)
def test_lamb_optimizer(
    dtype_x_lr,
    beta1_n_beta2_n_epsilon_n_lambda,
    mtr,
    inplace,
    stop_gradients,
    on_device,
    num_positional_args_init: pf.NumPositionalArg,
    num_positional_args_method: pf.NumPositionalArg,
    method_as_variable_flags: pf.AsVariableFlags,
    method_native_array_flags: pf.NativeArrayFlags,
    method_container_flags: pf.ContainerFlags,
    test_gradients: pf.BuiltGradientStrategy,
    class_name,
    method_name,
    ground_truth_backend,
):
    input_dtype, x, lr = dtype_x_lr
    beta1, beta2, epsilon, decay_lambda = beta1_n_beta2_n_epsilon_n_lambda
    xs_grad_idxs = [[0, 0]] if num_positional_args_method else [[1, "v"]]
    helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_num_positional_args=num_positional_args_init,
        init_all_as_kwargs_np={
            "lr": lr,
            "beta1": beta1,
            "beta2": beta2,
            "epsilon": epsilon,
            "max_trust_ratio": mtr,
            "decay_lambda": decay_lambda,
            "inplace": inplace,
            "stop_gradients": stop_gradients,
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=method_as_variable_flags,
        method_num_positional_args=num_positional_args_method,
        method_native_array_flags=method_native_array_flags,
        method_container_flags=method_container_flags,
        method_all_as_kwargs_np={
            "v": x[0],
            "grads": x[1],
        },
        class_name=class_name,
        method_name=method_name,
        rtol_=1e-1,
        atol_=1e-1,
        test_gradients=test_gradients,
        xs_grad_idxs=xs_grad_idxs,
        device_=on_device,
    )
