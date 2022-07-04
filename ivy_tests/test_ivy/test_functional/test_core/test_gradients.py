"""Collection of tests for unified gradient functions."""

# global
from numbers import Number
import pytest
import numpy as np
import torch
import tensorflow as tf
from hypothesis import given, strategies as st

# local
import ivy
import ivy.functional.backends.numpy
from ivy.functional.ivy.creation import native_array
import ivy_tests.test_ivy.helpers as helpers
from ivy.container import Container
import ivy.functional.backends.numpy as ivy_np


# variable   
@given(
    dtype_and_x=helpers.dtype_and_values(ivy_np.valid_float_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    native_array=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="variable"),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_variable(
    dtype_and_x,
    as_variable,
    with_out,
    native_array,
    num_positional_args,
    container,
    instance_method,
    fw
):
    dtype, x = dtype_and_x
    x = np.asarray(x, dtype=dtype)
    if x.shape == ():
        return
    helpers.test_function(
        dtype,
        as_variable,
        with_out,
        native_array,
        fw,
        num_positional_args,
        container,
        instance_method,
        "variable",
        x=x
    )


# is_variable
@given(
    dtype_and_x=helpers.dtype_and_values(ivy_np.valid_float_dtypes),
    as_variable=st.booleans(),
    native_array=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="is_variable"),
    container=st.booleans(),
    instance_method=st.booleans(),
    exclusive=st.booleans()
)
def test_is_variable(
    dtype_and_x,
    as_variable,
    native_array,
    num_positional_args,
    container,
    instance_method,
    fw,
    exclusive
):
    dtype, x = dtype_and_x
    x = np.asarray(x,dtype=dtype)
    helpers.test_function(
        dtype,
        as_variable,
        False,
        native_array,
        fw,
        num_positional_args,
        container,
        instance_method,
        "is_variable",
        x=x,
        exclusive=exclusive
    )

# variable data
@given(
    dtype_and_x=helpers.dtype_and_values(ivy_np.valid_float_dtypes),
    as_variable=st.booleans(),
    native_array=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="variable_data"),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_variable_data(
    dtype_and_x,
    as_variable,
    native_array,
    num_positional_args,
    container,
    instance_method,
    fw
):
    dtype, x = dtype_and_x
    x = np.asarray(x, dtype=dtype)
    if fw == "torch":
        x = torch.as_tensor(x,dtype=torch.float32)
    if fw == "tensorflow":
        x = tf.Variable(tf.convert_to_tensor(x,dtype=dtype))
    helpers.test_function(
        dtype,
        as_variable,
        False,
        native_array,
        fw,
        num_positional_args,
        container,
        instance_method,
        "variable_data",
        x=x
    )

# stop_gradient
@given(
    dtype_and_x=helpers.dtype_and_values(ivy_np.valid_float_dtypes),
    as_variable=st.booleans(),
    native_array=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="stop_gradient"),
    container=st.booleans(),
    instance_method=st.booleans(),
    preserve_type=st.booleans()
)
def test_stop_gradient(
    dtype_and_x,
    as_variable,
    native_array,
    num_positional_args,
    container,
    instance_method,
    fw,
    preserve_type
):
    dtype, x = dtype_and_x
    x = np.asarray(x, dtype=dtype)
    helpers.test_function(
        dtype,
        as_variable,
        False,
        native_array,
        fw,
        num_positional_args,
        container,
        instance_method,
        "stop_gradient",
        x=x,
        preserve_type=preserve_type
    )

# execute_with_gradients
@given(
    func_n_xs_n_ty_n_te_n_tg=st.sampled_from(
        [
            (
            lambda xs_in: (xs_in["w"] * xs_in["w"])[0],
            Container({"w": [3.0]}),
            np.array(9.0),
            None,
            {"w": np.array([6.0])},
            ),
            (
            lambda xs_in: ((xs_in["w"] * xs_in["w"])[0], xs_in["w"] * 1.5),
            Container({"w": [3.0]}),
            np.array(9.0),
            np.array([4.5]),
            {"w": np.array([6.0])},
            ),
            (
            lambda xs_in: (xs_in["w1"] * xs_in["w2"])[0],
            Container({"w1": [3.0], "w2": [5.0]}),
            np.array(15.0),
            None,
            {"w1": np.array([5.0]), "w2": np.array([3.0])},
            ),
        ]
    ),
    dtype=st.sampled_from(list(ivy_np.valid_float_dtypes) + [None]),
    tensor_fn=st.sampled_from([ivy.array, helpers.var_fn])
)
def test_execute_with_gradients(
    func_n_xs_n_ty_n_te_n_tg, dtype, tensor_fn, device, compile_graph, call
):
    # smoke test
    func, xs_raw, true_y, true_extra, true_dydxs = func_n_xs_n_ty_n_te_n_tg
    xs = xs_raw.map(lambda x, _: ivy.variable(ivy.array(x)))
    grad_fn = lambda xs_: ivy.execute_with_gradients(func, xs_)
    # TODO compile if this mode is set when ivy.compile is implemented
    if true_extra is None:
        y, dydxs = grad_fn(xs)
        extra_out = None
    else:
        y, dydxs, extra_out = grad_fn(xs)
    # type test
    assert ivy.is_array(y) or isinstance(y, Number)
    if call is not helpers.np_call:
        assert isinstance(dydxs, dict)
    # cardinality test
    if call is not helpers.mx_call:
        # mxnet cannot slice array down to shape (), it remains fixed at size (1,)
        assert y.shape == true_y.shape
    if call is not helpers.np_call:
        for (g, g_true) in zip(dydxs.values(), true_dydxs.values()):
            assert g.shape == g_true.shape
    # value test
    xs = xs_raw.map(lambda x, _: ivy.variable(ivy.array(x)))
    if true_extra is None:
        y, dydxs = call(ivy.execute_with_gradients, func, xs)
    else:
        y, dydxs, extra_out = call(ivy.execute_with_gradients, func, xs)
    assert np.allclose(y, true_y)
    if true_extra:
        assert np.allclose(extra_out, true_extra)
    if call is helpers.np_call:
        # numpy doesn't support autodiff
        assert dydxs is None
    else:
        for (g, g_true) in zip(dydxs.values(), true_dydxs.values()):
            assert np.allclose(ivy.to_numpy(g), g_true)

# adam_step
@given(
    dtype_and_dcdw=helpers.dtype_and_values(ivy_np.valid_float_dtypes),
    as_variable=st.booleans(),
    native_array=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="adam_step"),
    container=helpers.list_of_length(st.booleans(), 3),
    instance_method=st.booleans(),
    mw=st.floats(allow_infinity=False,allow_nan=False),
    vw=st.floats(allow_infinity=False,allow_nan=False),
    step=st.integers(min_value=1,max_value=1000).filter(lambda x: x > 0),
    beta1=st.floats(min_value=0.0,max_value=1.0,allow_nan=False).filter(lambda x: x != 0),
    beta2=st.floats(min_value=0.0,max_value=1.0,allow_nan=False).filter(lambda x: x != 0),
    epsilon=st.floats(min_value=0.0,max_value=1.0,allow_nan=False).filter(lambda x: x != 0),
)
def test_adam_step(
    dtype_and_dcdw,
    as_variable,
    native_array,
    num_positional_args,
    container,
    instance_method,
    fw,
    mw,
    vw,
    step,
    beta1,
    beta2,
    epsilon,
):
    dtype, dcdw = dtype_and_dcdw
    dcdw = np.asarray(dcdw,dtype=dtype)
    mw = np.asarray(mw,dtype=dtype)
    vw = np.asarray(vw,dtype=dtype)
    step = np.asarray(step,dtype=dtype)
    helpers.test_function(
        dtype,
        as_variable,
        False,
        native_array,
        fw,
        num_positional_args,
        container,
        instance_method,
        "adam_step",
        dcdw=dcdw,
        mw=mw,
        vw=vw,
        step=step,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
    )

# optimizer_update
@given(
    dtype_and_w=helpers.dtype_and_values(ivy_np.valid_float_dtypes),
    as_variable=st.booleans(),
    native_array=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="optimizer_update"),
    container=helpers.list_of_length(st.booleans(), 2),
    instance_method=st.booleans(),
    effective_grad=st.floats(),
    lr=st.floats(min_value=0.0,max_value=1.0, allow_nan=False),
    inplace=st.booleans(),
    stop_gradients=st.booleans()
)
def test_optimizer_update(
    dtype_and_w,
    as_variable,
    native_array,
    num_positional_args,
    container,
    instance_method,
    fw,
    effective_grad,
    lr,
    inplace,
    stop_gradients
):
    dtype, w = dtype_and_w
    w = np.asarray(w,dtype=dtype)
    effective_grad = np.asarray(effective_grad,dtype=dtype)
    helpers.test_function(
        dtype,
        as_variable,
        False,
        native_array,
        fw,
        num_positional_args,
        container,
        instance_method,
        "optimizer_update",
        w=w,
        effective_grad=effective_grad,
        lr=lr,
        inplace=inplace,
        stop_gradients=stop_gradients,
    )
    
# gradient_descent_update
@given(
    dtype_and_w=helpers.dtype_and_values(ivy_np.valid_float_dtypes),
    as_variable=st.booleans(),
    native_array=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="gradient_descent_update"),
    container=helpers.list_of_length(st.booleans(), 2),
    instance_method=st.booleans(),
    dcdw=st.floats(),
    lr=st.floats(min_value=0.0,max_value=1.0,allow_nan=False),
    inplace=st.booleans(),
    stop_gradients=st.booleans()
)
def test_gradient_descent_update(
    dtype_and_w,
    as_variable,
    native_array,
    num_positional_args,
    container,
    instance_method,
    fw,
    dcdw,
    lr,
    inplace,
    stop_gradients
):
    dtype, w = dtype_and_w
    w = np.asarray(w,dtype=dtype)
    dcdw = np.asarray(dcdw,dtype=dtype)
    helpers.test_function(
        dtype,
        as_variable,
        False,
        native_array,
        fw,
        num_positional_args,
        container,
        instance_method,
        "gradient_descent_update",
        w=w,
        dcdw=dcdw,
        lr=lr,
        inplace=inplace,
        stop_gradients=stop_gradients,
    )

# layerwise_gradient_descent_update
@given(
    ws_n_grads_n_lr_n_wsnew=st.sampled_from(
    [
        (
            Container({"a": [3.0], "b": [3.0]}),
            Container({"a": [6.0], "b": [6.0]}),
            Container({"a": [0.1], "b": [0.2]}),
            Container({"a": [2.4], "b": [1.8]}),
        )
    ],
    ),
    dtype=st.sampled_from(list(ivy_np.valid_float_dtypes) + [None]),
    tensor_fn=st.sampled_from([ivy.array, helpers.var_fn])
)
def test_layerwise_gradient_descent_update(
    ws_n_grads_n_lr_n_wsnew, dtype, tensor_fn, device, call
):
    # smoke test
    ws_raw, dcdw_raw, lr_raw, ws_raw_new = ws_n_grads_n_lr_n_wsnew
    ws = ws_raw.map(lambda x, _: ivy.variable(ivy.array(x)))
    dcdw = dcdw_raw.map(lambda x, _: ivy.array(x))
    lr = lr_raw.map(lambda x, _: ivy.array(x))
    ws_true_new = ws_raw_new.map(lambda x, _: ivy.variable(ivy.array(x)))
    ws_new = ivy.gradient_descent_update(ws, dcdw, lr)
    # type test
    assert isinstance(ws_new, dict)
    # cardinality test
    for (w_new, w_true_new) in zip(ws_new.values(), ws_true_new.values()):
        assert w_new.shape == w_true_new.shape
    # value test
    for (w_new, w_true_new) in zip(ws_new.values(), ws_true_new.values()):
        assert np.allclose(ivy.to_numpy(w_new), ivy.to_numpy(w_true_new))
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support internal function definitions
        return


# lars_update
@given(
    dtype_and_w=helpers.dtype_and_values(ivy_np.valid_float_dtypes),
    as_variable=st.booleans(),
    native_array=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="lars_update"),
    container=helpers.list_of_length(st.booleans(), 2),
    instance_method=st.booleans(),
    dcdw=st.floats(allow_infinity=False,allow_nan=False),
    lr=st.floats(min_value=0.0,max_value=1.0,allow_nan=False),
    decay_lambda=st.floats(min_value=0.0,max_value=1.0,allow_nan=False),
    inplace=st.booleans(),
    stop_gradients=st.booleans()
)
def test_lars_update(
    dtype_and_w,
    as_variable,
    native_array,
    num_positional_args,
    container,
    instance_method,
    fw,
    dcdw,
    lr,
    decay_lambda,
    inplace,
    stop_gradients
):
    dtype, w = dtype_and_w
    w = np.asarray(w,dtype=dtype)
    dcdw = np.asarray(dcdw,dtype=dtype)
    if fw == "torch":
        w = torch.as_tensor(w)
        dcdw = torch.as_tensor(dcdw)
    helpers.test_function(
        dtype,
        as_variable,
        False,
        native_array,
        fw,
        num_positional_args,
        container,
        instance_method,
        "lars_update",
        w=w,
        dcdw=dcdw,
        lr=lr,
        decay_lambda=decay_lambda,
        inplace=inplace,
        stop_gradients=stop_gradients,
    )

# adam_update
@given(
    ws_n_grads_n_lr_n_wsnew=st.sampled_from(
    [
        (
            Container({"w": [3.0]}),
            Container({"w": [6.0]}),
            0.1,
            Container({"w": [2.96837726]}),
        )
    ],
    ),
    dtype=st.sampled_from(list(ivy_np.valid_float_dtypes) + [None]),
    tensor_fn=st.sampled_from([ivy.array, helpers.var_fn])
)
def test_adam_update(ws_n_grads_n_lr_n_wsnew, dtype, tensor_fn, device, call):
    # smoke test
    ws_raw, dcdw_raw, lr, ws_raw_new = ws_n_grads_n_lr_n_wsnew
    ws = ws_raw.map(lambda x, _: ivy.variable(ivy.array(x)))
    dcdw = dcdw_raw.map(lambda x, _: ivy.array(x))
    ws_true_new = ws_raw_new.map(lambda x, _: ivy.variable(ivy.array(x)))
    mw = dcdw
    vw = dcdw.map(lambda x, _: x**2)
    ret = ivy.adam_update(ws, dcdw, lr, mw, vw, ivy.array(1))
    ws_new = {"ws_new": list(ret.values())[0][0]}
    mw_new = {"mw_new": list(ret.values())[0][1]}
    vw_new = {"vw_new": list(ret.values())[0][2]}
    # type test
    assert isinstance(ws_new, dict)
    assert isinstance(mw_new, dict)
    assert isinstance(vw_new, dict)
    # cardinality test
    for (w_new, w_true_new) in zip(ws_new.values(), ws_true_new.values()):
        assert w_new.shape == w_true_new.shape
    for (m_new, m_orig) in zip(mw_new.values(), mw.values()):
        assert m_new.shape == m_orig.shape
    for (v_new, v_orig) in zip(vw_new.values(), vw.values()):
        assert v_new.shape == v_orig.shape
    # value test
    for (w_new, w_true_new) in zip(ws_new.values(), ws_true_new.values()):
        assert np.allclose(ivy.to_numpy(w_new), ivy.to_numpy(w_true_new))
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support internal function definitions
        return
    
#adam_update
@given(
    dtype_and_w=helpers.dtype_and_values(ivy_np.valid_float_dtypes),
    as_variable=st.booleans(),
    native_array=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="adam_update"),
    container=helpers.list_of_length(st.booleans(), 4),
    instance_method=st.booleans(),
    dcdw=st.floats(allow_infinity=False,allow_nan=False),
    lr=st.floats(min_value=0.0,max_value=1.0,allow_nan=False),
    mw_tm1=st.floats(allow_infinity=False,allow_nan=False),
    vw_tm1=st.floats(allow_infinity=False,allow_nan=False),
    step=st.floats(allow_infinity=False,allow_nan=False),
    beta1=st.floats(min_value=0.0,max_value=1.0,allow_nan=False).filter(lambda x: x != 0),
    beta2=st.floats(min_value=0.0,max_value=1.0,allow_nan=False).filter(lambda x: x != 0),
    epsilon=st.floats(min_value=0.0,max_value=1.0,allow_nan=False).filter(lambda x: x != 0),
    inplace=st.booleans(),
    stop_gradients=st.booleans()
)
def test_adam_update(
    dtype_and_w,
    as_variable,
    native_array,
    num_positional_args,
    container,
    instance_method,
    fw,
    dcdw,
    lr,
    mw_tm1,
    vw_tm1,
    step,
    beta1,
    beta2,
    epsilon,
    inplace,
    stop_gradients
):
    dtype, w = dtype_and_w
    w = np.asarray(w,dtype=dtype)
    dcdw = np.asarray(dcdw,dtype=dtype)
    mw_tm1 = np.asarray(mw_tm1,dtype=dtype)
    vw_tm1 = np.asarray(vw_tm1,dtype=dtype)
    step = np.asarray(step,dtype=dtype)
    helpers.test_function(
        dtype,
        as_variable,
        False,
        native_array,
        fw,
        num_positional_args,
        container,
        instance_method,
        "adam_update",
        w=w,
        dcdw=dcdw,
        lr=lr,
        mw_tm1=mw_tm1,
        vw_tm1=vw_tm1,
        step=step,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        inplace=inplace,
        stop_gradients=stop_gradients
    )


# layerwise_adam_update
@given(
    ws_n_grads_n_lr_n_wsnew=st.sampled_from(
    [
        (
            Container({"a": [3.0], "b": [3.0]}),
            Container({"a": [6.0], "b": [6.0]}),
            Container({"a": [0.1], "b": [0.2]}),
            Container({"a": [2.9683773], "b": [2.9367545]}),
        )
    ],
    ),
    dtype=st.sampled_from(list(ivy_np.valid_float_dtypes) + [None]),
    tensor_fn=st.sampled_from([ivy.array, helpers.var_fn])
)
def test_layerwise_adam_update(ws_n_grads_n_lr_n_wsnew, dtype, tensor_fn, device, call):
    # smoke test
    ws_raw, dcdw_raw, lr_raw, ws_raw_new = ws_n_grads_n_lr_n_wsnew
    ws = ws_raw.map(lambda x, _: ivy.variable(ivy.array(x)))
    dcdw = dcdw_raw.map(lambda x, _: ivy.array(x))
    lr = lr_raw.map(lambda x, _: ivy.array(x))
    ws_true_new = ws_raw_new.map(lambda x, _: ivy.variable(ivy.array(x)))
    mw = dcdw
    vw = dcdw.map(lambda x, _: x**2)
    ws_new, mw_new, vw_new = ivy.adam_update(ws, dcdw, lr, mw, vw, ivy.array(1))
    # type test
    assert isinstance(ws_new, dict)
    assert isinstance(mw_new, dict)
    assert isinstance(vw_new, dict)
    # cardinality test
    for (w_new, w_true_new) in zip(ws_new.values(), ws_true_new.values()):
        assert w_new.shape == w_true_new.shape
    for (m_new, m_orig) in zip(mw_new.values(), mw.values()):
        assert m_new.shape == m_orig.shape
    for (v_new, v_orig) in zip(vw_new.values(), vw.values()):
        assert v_new.shape == v_orig.shape
    # value test
    for (w_new, w_true_new) in zip(ws_new.values(), ws_true_new.values()):
        assert np.allclose(ivy.to_numpy(w_new), ivy.to_numpy(w_true_new))
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support internal function definitions
        return


# lamb_update
@given(
    ws_n_grads_n_lr_n_wsnew=st.sampled_from(
    [
        (
            Container({"a": [3.0], "b": [3.0]}),
            Container({"a": [6.0], "b": [6.0]}),
            Container({"a": [0.1], "b": [0.2]}),
            Container({"a": [2.7], "b": [2.4]}),
        )
    ],
    ),
    dtype=st.sampled_from(list(ivy_np.valid_float_dtypes) + [None]),
    tensor_fn=st.sampled_from([ivy.array, helpers.var_fn])
)
def test_lamb_update(ws_n_grads_n_lr_n_wsnew, dtype, tensor_fn, device, call):
    # smoke test
    ws_raw, dcdw_raw, lr_raw, ws_raw_new = ws_n_grads_n_lr_n_wsnew
    ws = ws_raw.map(lambda x, _: ivy.variable(ivy.array(x)))
    dcdw = dcdw_raw.map(lambda x, _: ivy.array(x))
    lr = lr_raw.map(lambda x, _: ivy.array(x))
    ws_true_new = ws_raw_new.map(lambda x, _: ivy.variable(ivy.array(x)))
    mw = dcdw
    vw = dcdw.map(lambda x, _: x**2)
    ws_new,mw_new,vw_new = ivy.lamb_update(ws, dcdw, lr, mw, vw, ivy.array(1))
    # type test
    assert isinstance(ws_new, dict)
    assert isinstance(mw_new, dict)
    assert isinstance(vw_new, dict)
    # cardinality test
    for (w_new, w_true_new) in zip(ws_new.values(), ws_true_new.values()):
        assert w_new.shape == w_true_new.shape
    for (m_new, m_orig) in zip(mw_new.values(), mw.values()):
        assert m_new.shape == m_orig.shape
    for (v_new, v_orig) in zip(vw_new.values(), vw.values()):
        assert v_new.shape == v_orig.shape
    # value test
    for (w_new, w_true_new) in zip(ws_new.values(), ws_true_new.values()):
        assert np.allclose(ivy.to_numpy(w_new), ivy.to_numpy(w_true_new))
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support internal function definitions
        return

@given(
    dtype=st.sampled_from(ivy_np.valid_float_dtypes[1:]),
    as_variable=st.booleans(),
    native_array=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="lamb_update"),
    container=helpers.list_of_length(st.booleans(), 5),
    instance_method=st.booleans(),
    w=st.floats(allow_infinity=False,allow_nan=False).filter(lambda x: x != 0),
    dcdw=st.floats(allow_infinity=False,allow_nan=False),
    lr=st.floats(min_value=0.0,max_value=1.0,allow_nan=False),
    mw_tm1=st.floats(allow_infinity=False,allow_nan=False),
    vw_tm1=st.floats(allow_infinity=False,allow_nan=False),
    step=st.integers(min_value=1,max_value=1000).filter(lambda x: x > 0),
    beta1=st.floats(min_value=0.0,max_value=1.0,allow_nan=False).filter(lambda x: x != 0),
    beta2=st.floats(min_value=0.0,max_value=1.0,allow_nan=False).filter(lambda x: x != 0),
    epsilon=st.floats(min_value=0.0,max_value=1.0,allow_nan=False).filter(lambda x: x != 0),
    max_trust_ratio=st.floats(allow_infinity=False,allow_nan=False).filter(lambda x: x > 0),
    decay_lambda=st.floats(min_value=0.0,max_value=1.0,allow_nan=False),
    inplace=st.booleans(),
    stop_gradients=st.booleans()
)
def test_lamb_update(
    dtype,
    as_variable,
    native_array,
    num_positional_args,
    container,
    instance_method,
    fw,
    w,
    dcdw,
    lr,
    mw_tm1,
    vw_tm1,
    step,
    beta1,
    beta2,
    epsilon,
    max_trust_ratio,
    decay_lambda,
    inplace,
    stop_gradients
):
    w = np.asarray(w,dtype=dtype)
    dcdw = np.asarray(dcdw,dtype=dtype)
    lr = np.asarray(lr,dtype=dtype)
    mw_tm1 = np.asarray(mw_tm1,dtype=dtype)
    vw_tm1 = np.asarray(vw_tm1,dtype=dtype)
    step = np.asarray(step,dtype=dtype)
    if fw == "torch":
        w = torch.as_tensor(w)
        dcdw = torch.as_tensor(dcdw)
        lr = torch.as_tensor(lr)
        mw_tm1 = torch.as_tensor(mw_tm1)
        vw_tm1 = torch.as_tensor(vw_tm1)
    helpers.test_function(
        dtype,
        as_variable,
        False,
        native_array,
        fw,
        num_positional_args,
        False,
        instance_method,
        "lamb_update",
        w=w,
        dcdw=dcdw,
        lr=lr,
        mw_tm1=mw_tm1,
        vw_tm1=vw_tm1,
        step=step,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        max_trust_ratio=max_trust_ratio,
        decay_lambda=decay_lambda,
        inplace=inplace,
        stop_gradients=stop_gradients
    )

# Still to Add #
# ---------------#

# with_grads
# set_with_grads
# unset_with_grads
# execute_with_gradients
# adam_step
