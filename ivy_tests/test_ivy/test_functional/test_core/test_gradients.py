"""Collection of tests for unified gradient functions."""

# global
from hypothesis import given, strategies as st
import pytest
import numpy as np

# local
import ivy
import ivy.functional.backends.numpy as ivy_np
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@st.composite
def get_gradient_arguments_with_lr(draw, *, num_arrays=1, no_lr=False):
    dtypes, arrays, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=ivy_np.valid_float_dtypes,
            num_arrays=num_arrays,
            min_value=1,
            safety_factor=0.8,
            exclude_min=True,
            min_num_dims=1,
            shared_dtype=True,
            ret_shape=True,
        )
    )
    dtype = dtypes[0]
    if no_lr:
        return dtypes, arrays
    lr = draw(
        st.one_of(
            st.floats(min_value=0.0, max_value=1.0, exclude_min=True, width=32),
            helpers.array_values(
                dtype=dtype, shape=shape, min_value=0.0, exclude_min=True
            ),
        )
    )
    if isinstance(lr, list):
        dtypes += [dtype]
    return dtypes, arrays, lr


@pytest.mark.parametrize("grads", [True, False])
def test_with_grads(grads):
    assert ivy.with_grads(with_grads=grads) == grads


@pytest.mark.parametrize("grads", [True, False])
def test_set_with_grads(grads):
    ivy.set_with_grads(grads)
    assert ivy.with_grads(with_grads=None) == grads


@pytest.mark.parametrize("grads", [True, False])
def test_unset_with_grads(grads):
    ivy.set_with_grads(grads)
    with_grad_stack = ivy.with_grads_stack.copy()
    ivy.unset_with_grads()
    assert with_grad_stack[1:] == ivy.with_grads_stack


# variable
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_float_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="variable"),
    data=st.data(),
)
@handle_cmd_line_args
def test_variable(
    *,
    dtype_and_x,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=True,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="variable",
        x=np.asarray(x, dtype=dtype),
    )


# is_variable
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_float_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="is_variable"),
    data=st.data(),
)
@handle_cmd_line_args
def test_is_variable(
    *,
    dtype_and_x,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=True,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="is_variable",
        x=np.asarray(x, dtype=dtype),
    )


# variable data
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_float_dtypes,
    ),
    data=st.data(),
)
@handle_cmd_line_args
def test_variable_data(dtype_and_x, native_array, container, instance_method, fw):
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        with_out=False,
        as_variable_flags=True,
        num_positional_args=1,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="variable_data",
        x=np.asarray(x, dtype=dtype),
    )


# stop_gradient
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_float_dtypes,
    ),
    preserve_type=st.booleans(),
    data=st.data(),
)
@handle_cmd_line_args
def test_stop_gradient(
    dtype_and_x, preserve_type, with_out, native_array, container, instance_method, fw
):
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        with_out=with_out,
        as_variable_flags=True,
        num_positional_args=1,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="stop_gradient",
        x=np.asarray(x, dtype=dtype),
        preserve_type=preserve_type,
    )


# execute_with_gradients
@given(
    dtype_and_xs=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_float_dtypes, min_num_dims=1, min_dim_size=1
    ),
    retain_grads=st.booleans(),
    container_flag=st.booleans(),
    data=st.data(),
)
@handle_cmd_line_args
def test_execute_with_gradients(
    *,
    dtype_and_xs,
    retain_grads,
    container_flag,
    native_array,
    fw,
):
    def func(xs):
        array_idxs = ivy.nested_indices_where(xs, ivy.is_ivy_array)
        array_vals = ivy.multi_index_nest(xs, array_idxs)
        final_array = ivy.stack(array_vals)
        ret = ivy.sum(final_array)
        return ret

    dtype, xs = dtype_and_xs
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=True,
        with_out=False,
        num_positional_args=2,
        native_array_flags=native_array,
        container_flags=container_flag,
        instance_method=False,
        fw=fw,
        fn_name="execute_with_gradients",
        func=func,
        xs=np.asarray(xs, dtype=dtype),
        retain_grads=retain_grads,
    )


# adam_step
@given(
    dtype_n_dcdw_n_mw_n_vw=get_gradient_arguments_with_lr(num_arrays=3, no_lr=True),
    step=st.integers(min_value=1, max_value=100),
    beta1_n_beta2_n_epsilon=helpers.lists(
        arg=st.floats(min_value=0, max_value=1, exclude_min=True, width=32),
        min_size=3,
        max_size=3,
    ),
    data=st.data(),
)
@handle_cmd_line_args
def test_adam_step(
    *,
    dtype_n_dcdw_n_mw_n_vw,
    step,
    beta1_n_beta2_n_epsilon,
    as_variable,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtypes, [dcdw, mw, vw] = dtype_n_dcdw_n_mw_n_vw
    (
        beta1,
        beta2,
        epsilon,
    ) = beta1_n_beta2_n_epsilon
    helpers.test_function(
        input_dtypes=input_dtypes,
        with_out=False,
        as_variable_flags=as_variable,
        num_positional_args=4,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="adam_step",
        dcdw=np.asarray(dcdw, dtype=input_dtypes[0]),
        mw=np.asarray(mw, input_dtypes[1]),
        vw=np.asarray(vw, dtype=input_dtypes[2]),
        step=step,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
    )


# optimizer_update
@given(
    dtype_n_ws_n_effgrad_n_lr=get_gradient_arguments_with_lr(num_arrays=2),
    stop_gradients=st.booleans(),
    data=st.data(),
)
@handle_cmd_line_args
def test_optimizer_update(
    dtype_n_ws_n_effgrad_n_lr,
    stop_gradients,
    with_out,
    as_variable,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtypes, [w, effective_grad], lr = dtype_n_ws_n_effgrad_n_lr
    helpers.test_function(
        input_dtypes=input_dtypes,
        with_out=with_out,
        as_variable_flags=as_variable,
        num_positional_args=3,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="optimizer_update",
        w=np.asarray(w, dtype=input_dtypes[0]),
        effective_grad=np.asarray(effective_grad, dtype=input_dtypes[1]),
        lr=lr if isinstance(lr, float) else np.asarray(lr, dtype=input_dtypes[0]),
        stop_gradients=stop_gradients,
    )


# gradient_descent_update
@given(
    dtype_n_ws_n_dcdw_n_lr=get_gradient_arguments_with_lr(num_arrays=2),
    stop_gradients=st.booleans(),
    data=st.data(),
)
@handle_cmd_line_args
def test_gradient_descent_update(
    *,
    dtype_n_ws_n_dcdw_n_lr,
    stop_gradients,
    with_out,
    as_variable,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtypes, [w, dcdw], lr = dtype_n_ws_n_dcdw_n_lr
    helpers.test_function(
        input_dtypes=input_dtypes,
        with_out=with_out,
        as_variable_flags=as_variable,
        num_positional_args=3,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="gradient_descent_update",
        w=np.asarray(w, dtype=input_dtypes[0]),
        dcdw=np.asarray(dcdw, dtype=input_dtypes[1]),
        lr=lr if isinstance(lr, float) else np.asarray(lr, dtype=input_dtypes[0]),
        stop_gradients=stop_gradients,
    )


# lars_update
@given(
    dtype_n_ws_n_dcdw_n_lr=get_gradient_arguments_with_lr(num_arrays=2),
    decay_lambda=st.floats(min_value=0, max_value=1, exclude_min=True, width=32),
    stop_gradients=st.booleans(),
    data=st.data(),
)
@handle_cmd_line_args
def test_lars_update(
    *,
    dtype_n_ws_n_dcdw_n_lr,
    decay_lambda,
    stop_gradients,
    as_variable,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtypes, [w, dcdw], lr = dtype_n_ws_n_dcdw_n_lr
    helpers.test_function(
        input_dtypes=input_dtypes,
        with_out=False,
        as_variable_flags=as_variable,
        num_positional_args=3,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="lars_update",
        w=np.asarray(w, dtype=input_dtypes[0]),
        dcdw=np.asarray(dcdw, dtype=input_dtypes[1]),
        lr=lr if isinstance(lr, float) else np.asarray(lr, dtype=input_dtypes[0]),
        decay_lambda=decay_lambda,
        stop_gradients=stop_gradients,
    )


# adam_update
@given(
    dtype_n_ws_n_dcdw_n_mwtm1_n_vwtm1_n_lr=get_gradient_arguments_with_lr(num_arrays=4),
    step=st.integers(min_value=1, max_value=100),
    beta1_n_beta2_n_epsilon=helpers.lists(
        arg=st.floats(min_value=0, max_value=1, exclude_min=True, width=32),
        min_size=3,
        max_size=3,
    ),
    stopgrad=st.booleans(),
    data=st.data(),
)
@handle_cmd_line_args
def test_adam_update(
    *,
    dtype_n_ws_n_dcdw_n_mwtm1_n_vwtm1_n_lr,
    step,
    beta1_n_beta2_n_epsilon,
    stopgrad,
    as_variable,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtypes, [w, dcdw, mw_tm1, vw_tm1], lr = dtype_n_ws_n_dcdw_n_mwtm1_n_vwtm1_n_lr
    (
        beta1,
        beta2,
        epsilon,
    ) = beta1_n_beta2_n_epsilon
    stop_gradients = stopgrad
    helpers.test_function(
        input_dtypes=input_dtypes,
        with_out=False,
        as_variable_flags=as_variable,
        num_positional_args=6,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="adam_update",
        w=np.asarray(w, dtype=input_dtypes[0]),
        dcdw=np.asarray(dcdw, dtype=input_dtypes[1]),
        lr=lr if isinstance(lr, float) else np.asarray(lr, dtype=input_dtypes[0]),
        mw_tm1=np.asarray(mw_tm1, input_dtypes[2]),
        vw_tm1=np.asarray(vw_tm1, dtype=input_dtypes[3]),
        step=step,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        stop_gradients=stop_gradients,
    )


# lamb_update
@given(
    dtype_n_ws_n_dcdw_n_mwtm1_n_vwtm1_n_lr=get_gradient_arguments_with_lr(num_arrays=4),
    step=st.integers(min_value=1, max_value=100),
    beta1_n_beta2_n_epsilon_n_lambda=helpers.lists(
        arg=st.floats(min_value=0, max_value=1, exclude_min=True, width=32),
        min_size=4,
        max_size=4,
    ),
    mtr=st.one_of(
        st.integers(min_value=1), st.floats(min_value=0, exclude_min=True, width=32)
    ),
    stopgrad=st.booleans(),
    data=st.data(),
)
@handle_cmd_line_args
def test_lamb_update(
    *,
    dtype_n_ws_n_dcdw_n_mwtm1_n_vwtm1_n_lr,
    step,
    beta1_n_beta2_n_epsilon_n_lambda,
    mtr,
    stopgrad,
    as_variable,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtypes, [w, dcdw, mw_tm1, vw_tm1], lr = dtype_n_ws_n_dcdw_n_mwtm1_n_vwtm1_n_lr
    (
        beta1,
        beta2,
        epsilon,
        decay_lambda,
    ) = beta1_n_beta2_n_epsilon_n_lambda
    max_trust_ratio, stop_gradients = mtr, stopgrad
    helpers.test_function(
        input_dtypes=input_dtypes,
        with_out=False,
        as_variable_flags=as_variable,
        num_positional_args=6,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="lamb_update",
        w=np.asarray(w, dtype=input_dtypes[0]),
        dcdw=np.asarray(dcdw, dtype=input_dtypes[1]),
        lr=lr if isinstance(lr, float) else np.asarray(lr, dtype=input_dtypes[0]),
        mw_tm1=np.asarray(mw_tm1, input_dtypes[2]),
        vw_tm1=np.asarray(vw_tm1, dtype=input_dtypes[3]),
        step=step,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        max_trust_ratio=max_trust_ratio,
        decay_lambda=decay_lambda,
        stop_gradients=stop_gradients,
    )


# Still to Add #
# ---------------#

# with_grads
# set_with_grads
# unset_with_grads
