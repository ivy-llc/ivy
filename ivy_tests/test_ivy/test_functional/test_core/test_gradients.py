"""Collection of tests for unified gradient functions."""

# global
from numbers import Number
import pytest
import numpy as np

# local
import ivy
import ivy.functional.backends.numpy
import ivy_tests.test_ivy.helpers as helpers
from ivy.container import Container


# variable
@pytest.mark.parametrize("object_in", [[], [0.0], [1], [True], [[1.0, 2.0]]])
@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_variable(object_in, dtype, device, call):
    if call is helpers.tf_graph_call:
        # cannot create variables as part of compiled tf graph
        pytest.skip()
    if call in [helpers.mx_call] and dtype == "int16":
        # mxnet does not support int16
        pytest.skip()
    if len(object_in) == 0 and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    # smoke test
    ret = ivy.variable(ivy.array(object_in, dtype=dtype, device=device))
    # type test
    if call is not helpers.np_call:
        assert ivy.is_variable(ret)
    # cardinality test
    assert ret.shape == np.array(object_in).shape
    # value test
    assert np.allclose(
        call(ivy.variable, ivy.array(object_in, dtype=dtype, device=device)),
        np.array(object_in).astype(dtype),
    )
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support string devices
        return


# is_variable
@pytest.mark.parametrize("object_in", [[], [0.0], [1], [True], [[1.0, 2.0]]])
@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_is_variable(object_in, dtype, device, call):
    if call is helpers.tf_graph_call:
        # cannot create variables as part of compiled tf graph
        pytest.skip()
    if call in [helpers.mx_call] and dtype == "int16":
        # mxnet does not support int16
        pytest.skip()
    if len(object_in) == 0 and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    # smoke test
    non_var = ivy.array(object_in, dtype=dtype, device=device)
    var = ivy.variable(ivy.array(object_in, dtype=dtype, device=device))
    non_var_res = ivy.is_variable(non_var)
    var_res = ivy.is_variable(var)
    # type test
    assert ivy.is_ivy_array(non_var)
    if call is not helpers.np_call:
        assert ivy.is_variable(var)
    if call in [helpers.np_call, helpers.jnp_call]:
        # numpy and jax do not support flagging variables
        pytest.skip()
    # value test
    assert non_var_res is False
    assert var_res is True


# variable data
@pytest.mark.parametrize("object_in", [[], [0.0], [1], [True], [[1.0, 2.0]]])
@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_variable_data(object_in, dtype, device, call):
    if call is helpers.tf_graph_call:
        # cannot create variables as part of compiled tf graph
        pytest.skip()
    if call in [helpers.mx_call] and dtype == "int16":
        # mxnet does not support int16
        pytest.skip()
    if len(object_in) == 0 and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    # smoke test
    var = ivy.variable(ivy.array(object_in, dtype=dtype, device=device))
    var_data = ivy.variable_data(var)
    # type test
    if call is not helpers.np_call:
        # numpy does not support variables
        assert ivy.is_variable(var)
        if call is not helpers.mx_call:
            # jax variables and their data are the same instance
            assert not ivy.is_variable(var_data, exclusive=True)
        assert ivy.is_ivy_array(var_data)
    # cardinality test
    assert var_data.shape == var.shape
    # value test
    assert np.allclose(ivy.to_numpy(var), ivy.to_numpy(var_data))


# stop_gradient
@pytest.mark.parametrize("x_raw", [[0.0]])
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [("array", ivy.array), ("var", helpers.var_fn)])
def test_stop_gradient(x_raw, dtype, tensor_fn, device, call):
    # smoke test
    fn_name, tensor_fn = tensor_fn
    x = tensor_fn(x_raw, dtype=dtype, device=device)
    ret = ivy.stop_gradient(x)
    # type test
    if fn_name == "array":
        assert ivy.is_ivy_array(ret)
    elif call is not helpers.np_call:
        # Numpy does not support variables, is_variable() always returns False
        assert ivy.is_variable(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    if call is not helpers.tf_graph_call:
        # Tf graph mode cannot create variables as part of the computation graph
        assert np.array_equal(
            call(ivy.stop_gradient, x),
            np.array(x_raw, dtype=dtype),
        )
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support attribute setting
        return


# execute_with_gradients
@pytest.mark.parametrize(
    "func_n_xs_n_ty_n_te_n_tg",
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
    ],
)
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array])
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


# gradient_descent_update
@pytest.mark.parametrize(
    "ws_n_grads_n_lr_n_wsnew",
    [(Container({"w": [3.0]}), Container({"w": [6.0]}), 0.1, Container({"w": [2.4]}))],
)
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_gradient_descent_update(
    ws_n_grads_n_lr_n_wsnew, dtype, tensor_fn, device, call
):
    # smoke test
    ws_raw, dcdws_raw, lr, ws_raw_new = ws_n_grads_n_lr_n_wsnew
    ws = ws_raw.map(lambda x, _: ivy.variable(ivy.array(x)))
    dcdws = dcdws_raw.map(lambda x, _: ivy.array(x))
    ws_true_new = ws_raw_new.map(lambda x, _: ivy.variable(ivy.array(x)))
    ws_new = ivy.gradient_descent_update(ws, dcdws, lr)
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


# layerwise_gradient_descent_update
@pytest.mark.parametrize(
    "ws_n_grads_n_lr_n_wsnew",
    [
        (
            Container({"a": [3.0], "b": [3.0]}),
            Container({"a": [6.0], "b": [6.0]}),
            Container({"a": [0.1], "b": [0.2]}),
            Container({"a": [2.4], "b": [1.8]}),
        )
    ],
)
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_layerwise_gradient_descent_update(
    ws_n_grads_n_lr_n_wsnew, dtype, tensor_fn, device, call
):
    # smoke test
    ws_raw, dcdws_raw, lr_raw, ws_raw_new = ws_n_grads_n_lr_n_wsnew
    ws = ws_raw.map(lambda x, _: ivy.variable(ivy.array(x)))
    dcdws = dcdws_raw.map(lambda x, _: ivy.array(x))
    lr = lr_raw.map(lambda x, _: ivy.array(x))
    ws_true_new = ws_raw_new.map(lambda x, _: ivy.variable(ivy.array(x)))
    ws_new = ivy.gradient_descent_update(ws, dcdws, lr)
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
@pytest.mark.parametrize(
    "ws_n_grads_n_lr_n_wsnew",
    [
        (
            Container({"a": [3.0], "b": [3.0]}),
            Container({"a": [6.0], "b": [6.0]}),
            Container({"a": [0.1], "b": [0.2]}),
            Container({"a": [2.7], "b": [2.4]}),
        )
    ],
)
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_lars_update(ws_n_grads_n_lr_n_wsnew, dtype, tensor_fn, device, call):
    # smoke test
    ws_raw, dcdws_raw, lr_raw, ws_raw_new = ws_n_grads_n_lr_n_wsnew
    ws = ws_raw.map(lambda x, _: ivy.variable(ivy.array(x)))
    dcdws = dcdws_raw.map(lambda x, _: ivy.array(x))
    lr = lr_raw.map(lambda x, _: ivy.array(x))
    ws_true_new = ws_raw_new.map(lambda x, _: ivy.variable(ivy.array(x)))
    ws_new = ivy.lars_update(ws, dcdws, lr)
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


# adam_update
@pytest.mark.parametrize(
    "ws_n_grads_n_lr_n_wsnew",
    [
        (
            Container({"w": [3.0]}),
            Container({"w": [6.0]}),
            0.1,
            Container({"w": [2.96837726]}),
        )
    ],
)
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_adam_update(ws_n_grads_n_lr_n_wsnew, dtype, tensor_fn, device, call):
    # smoke test
    ws_raw, dcdws_raw, lr, ws_raw_new = ws_n_grads_n_lr_n_wsnew
    ws = ws_raw.map(lambda x, _: ivy.variable(ivy.array(x)))
    dcdws = dcdws_raw.map(lambda x, _: ivy.array(x))
    ws_true_new = ws_raw_new.map(lambda x, _: ivy.variable(ivy.array(x)))
    mw = dcdws
    vw = dcdws.map(lambda x, _: x**2)
    ws_new, mw_new, vw_new = ivy.adam_update(ws, dcdws, lr, mw, vw, ivy.array(1))
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


# layerwise_adam_update
@pytest.mark.parametrize(
    "ws_n_grads_n_lr_n_wsnew",
    [
        (
            Container({"a": [3.0], "b": [3.0]}),
            Container({"a": [6.0], "b": [6.0]}),
            Container({"a": [0.1], "b": [0.2]}),
            Container({"a": [2.9683773], "b": [2.9367545]}),
        )
    ],
)
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_layerwise_adam_update(ws_n_grads_n_lr_n_wsnew, dtype, tensor_fn, device, call):
    # smoke test
    ws_raw, dcdws_raw, lr_raw, ws_raw_new = ws_n_grads_n_lr_n_wsnew
    ws = ws_raw.map(lambda x, _: ivy.variable(ivy.array(x)))
    dcdws = dcdws_raw.map(lambda x, _: ivy.array(x))
    lr = lr_raw.map(lambda x, _: ivy.array(x))
    ws_true_new = ws_raw_new.map(lambda x, _: ivy.variable(ivy.array(x)))
    mw = dcdws
    vw = dcdws.map(lambda x, _: x**2)
    ws_new, mw_new, vw_new = ivy.adam_update(ws, dcdws, lr, mw, vw, ivy.array(1))
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
@pytest.mark.parametrize(
    "ws_n_grads_n_lr_n_wsnew",
    [
        (
            Container({"a": [3.0], "b": [3.0]}),
            Container({"a": [6.0], "b": [6.0]}),
            Container({"a": [0.1], "b": [0.2]}),
            Container({"a": [2.7], "b": [2.4]}),
        )
    ],
)
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_lamb_update(ws_n_grads_n_lr_n_wsnew, dtype, tensor_fn, device, call):
    # smoke test
    ws_raw, dcdws_raw, lr_raw, ws_raw_new = ws_n_grads_n_lr_n_wsnew
    ws = ws_raw.map(lambda x, _: ivy.variable(ivy.array(x)))
    dcdws = dcdws_raw.map(lambda x, _: ivy.array(x))
    lr = lr_raw.map(lambda x, _: ivy.array(x))
    ws_true_new = ws_raw_new.map(lambda x, _: ivy.variable(ivy.array(x)))
    mw = dcdws
    vw = dcdws.map(lambda x, _: x**2)
    ws_new, mw_new, vw_new = ivy.lamb_update(ws, dcdws, lr, mw, vw, ivy.array(1))
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


# Still to Add #
# ---------------#

# with_grads
# set_with_grads
# unset_with_grads
# variable
# is_variable
# variable_data
# stop_gradient
# execute_with_gradients
# adam_step
# optimizer_update
