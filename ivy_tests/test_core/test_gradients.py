"""
Collection of tests for templated gradient functions
"""

# global
import pytest
import numpy as np
from numbers import Number

# local
import ivy
import ivy.numpy
import ivy_tests.helpers as helpers
from ivy.core.container import Container


# variable
@pytest.mark.parametrize(
    "object_in", [[], [0.], [1], [True], [[1., 2.]]])
@pytest.mark.parametrize(
    "dtype_str", ['float16', 'float32', 'float64'])
def test_variable(object_in, dtype_str, dev_str, call):
    if call is helpers.tf_graph_call:
        # cannot create variables as part of compiled tf graph
        pytest.skip()
    if call in [helpers.mx_call] and dtype_str == 'int16':
        # mxnet does not support int16
        pytest.skip()
    if len(object_in) == 0 and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    # smoke test
    ret = ivy.variable(ivy.array(object_in, dtype_str, dev_str))
    # type test
    if call is not helpers.np_call:
        assert ivy.is_variable(ret)
    # cardinality test
    assert ret.shape == np.array(object_in).shape
    # value test
    assert np.allclose(call(ivy.variable, ivy.array(object_in, dtype_str, dev_str)),
                       np.array(object_in).astype(dtype_str))
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support string devices
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.variable)


# is_variable
@pytest.mark.parametrize(
    "object_in", [[], [0.], [1], [True], [[1., 2.]]])
@pytest.mark.parametrize(
    "dtype_str", ['float16', 'float32', 'float64'])
def test_is_variable(object_in, dtype_str, dev_str, call):
    if call is helpers.tf_graph_call:
        # cannot create variables as part of compiled tf graph
        pytest.skip()
    if call in [helpers.mx_call] and dtype_str == 'int16':
        # mxnet does not support int16
        pytest.skip()
    if len(object_in) == 0 and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    # smoke test
    non_var = ivy.array(object_in, dtype_str, dev_str)
    var = ivy.variable(ivy.array(object_in, dtype_str, dev_str))
    non_var_res = ivy.is_variable(non_var)
    var_res = ivy.is_variable(var)
    # type test
    assert ivy.is_array(non_var)
    if call is not helpers.np_call:
        assert ivy.is_variable(var)
    if call in [helpers.np_call, helpers.jnp_call]:
        # numpy and jax do not support flagging variables
        pytest.skip()
    # value test
    assert non_var_res is False
    assert var_res is True
    # compilation test
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.is_variable)


# variable data
@pytest.mark.parametrize(
    "object_in", [[], [0.], [1], [True], [[1., 2.]]])
@pytest.mark.parametrize(
    "dtype_str", ['float16', 'float32', 'float64'])
def test_variable_data(object_in, dtype_str, dev_str, call):
    if call is helpers.tf_graph_call:
        # cannot create variables as part of compiled tf graph
        pytest.skip()
    if call in [helpers.mx_call] and dtype_str == 'int16':
        # mxnet does not support int16
        pytest.skip()
    if len(object_in) == 0 and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    # smoke test
    var = ivy.variable(ivy.array(object_in, dtype_str, dev_str))
    var_data = ivy.variable_data(var)
    # type test
    if call is not helpers.np_call:
        # numpy does not support variables
        assert ivy.is_variable(var)
        if call is not helpers.mx_call:
            # jax variables and their data are the same instance
            assert not ivy.is_variable(var_data, exclusive=True)
        assert ivy.is_array(var_data)
    # cardinality test
    assert var_data.shape == var.shape
    # value test
    assert np.allclose(ivy.to_numpy(var), ivy.to_numpy(var_data))


# stop_gradient
@pytest.mark.parametrize(
    "x_raw", [[0.]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [('array', ivy.array), ('var', helpers.var_fn)])
def test_stop_gradient(x_raw, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    fn_name, tensor_fn = tensor_fn
    x = tensor_fn(x_raw, dtype_str, dev_str)
    ret = ivy.stop_gradient(x)
    # type test
    if fn_name == 'array':
        assert ivy.is_array(ret)
    elif call is not helpers.np_call:
        # Numpy does not support variables, is_variable() always returns False
        assert ivy.is_variable(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    if call is not helpers.tf_graph_call:
        # Tf graph mode cannot create variables as part of the computation graph
        assert np.array_equal(call(ivy.stop_gradient, x), ivy.numpy.array(x_raw, dtype_str))
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support attribute setting
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.stop_gradient)


# execute_with_gradients
@pytest.mark.parametrize(
    "func_n_xs_n_ty_n_te_n_tg", [(lambda xs_in: (xs_in['w'] * xs_in['w'])[0],
                                  Container({'w': [3.]}), np.array(9.), None, {'w': np.array([6.])}),
                                 (lambda xs_in: ((xs_in['w'] * xs_in['w'])[0], xs_in['w'] * 1.5),
                                  Container({'w': [3.]}), np.array(9.), np.array([4.5]), {'w': np.array([6.])}),
                                 (lambda xs_in: (xs_in['w1'] * xs_in['w2'])[0],
                                  Container({'w1': [3.], 'w2': [5.]}), np.array(15.), None,
                                  {'w1': np.array([5.]), 'w2': np.array([3.])})])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array])
def test_execute_with_gradients(func_n_xs_n_ty_n_te_n_tg, dtype_str, tensor_fn, dev_str, compile_graph, call):
    # smoke test
    func, xs_raw, true_y, true_extra, true_dydxs = func_n_xs_n_ty_n_te_n_tg
    xs = xs_raw.map(lambda x, _: ivy.variable(ivy.array(x)))
    grad_fn = lambda xs_: ivy.execute_with_gradients(func, xs_)
    if compile_graph and call is helpers.torch_call:
        # Currently only PyTorch is supported for ivy compilation
        grad_fn = ivy.compile_graph(grad_fn, xs)
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
    "ws_n_grads_n_lr_n_wsnew", [(Container({'w': [3.]}), Container({'w': [6.]}), 0.1, Container({'w': [2.4]}))])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_gradient_descent_update(ws_n_grads_n_lr_n_wsnew, dtype_str, tensor_fn, dev_str, call):
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
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.gradient_descent_update)


# layerwise_gradient_descent_update
@pytest.mark.parametrize(
    "ws_n_grads_n_lr_n_wsnew", [(Container({'a': [3.], 'b': [3.]}), Container({'a': [6.], 'b': [6.]}),
                                 Container({'a': [0.1], 'b': [0.2]}), Container({'a': [2.4], 'b': [1.8]}))])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_layerwise_gradient_descent_update(ws_n_grads_n_lr_n_wsnew, dtype_str, tensor_fn, dev_str, call):
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
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.gradient_descent_update)


# lars_update
@pytest.mark.parametrize(
    "ws_n_grads_n_lr_n_wsnew", [(Container({'a': [3.], 'b': [3.]}), Container({'a': [6.], 'b': [6.]}),
                                 Container({'a': [0.1], 'b': [0.2]}), Container({'a': [2.7], 'b': [2.4]}))])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_lars_update(ws_n_grads_n_lr_n_wsnew, dtype_str, tensor_fn, dev_str, call):
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
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.lars_update)


# adam_update
@pytest.mark.parametrize(
    "ws_n_grads_n_lr_n_wsnew", [(Container({'w': [3.]}), Container({'w': [6.]}), 0.1, Container({'w': [2.96837726]}))])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_adam_update(ws_n_grads_n_lr_n_wsnew, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    ws_raw, dcdws_raw, lr, ws_raw_new = ws_n_grads_n_lr_n_wsnew
    ws = ws_raw.map(lambda x, _: ivy.variable(ivy.array(x)))
    dcdws = dcdws_raw.map(lambda x, _: ivy.array(x))
    ws_true_new = ws_raw_new.map(lambda x, _: ivy.variable(ivy.array(x)))
    mw = dcdws
    vw = dcdws.map(lambda x, _: x ** 2)
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
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.adam_update)


# layerwise_adam_update
@pytest.mark.parametrize(
    "ws_n_grads_n_lr_n_wsnew", [(Container({'a': [3.], 'b': [3.]}), Container({'a': [6.], 'b': [6.]}),
                                 Container({'a': [0.1], 'b': [0.2]}), Container({'a': [2.9683773], 'b': [2.9367545]}))])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_layerwise_adam_update(ws_n_grads_n_lr_n_wsnew, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    ws_raw, dcdws_raw, lr_raw, ws_raw_new = ws_n_grads_n_lr_n_wsnew
    ws = ws_raw.map(lambda x, _: ivy.variable(ivy.array(x)))
    dcdws = dcdws_raw.map(lambda x, _: ivy.array(x))
    lr = lr_raw.map(lambda x, _: ivy.array(x))
    ws_true_new = ws_raw_new.map(lambda x, _: ivy.variable(ivy.array(x)))
    mw = dcdws
    vw = dcdws.map(lambda x, _: x ** 2)
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
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.adam_update)


# lamb_update
@pytest.mark.parametrize(
    "ws_n_grads_n_lr_n_wsnew", [(Container({'a': [3.], 'b': [3.]}), Container({'a': [6.], 'b': [6.]}),
                                 Container({'a': [0.1], 'b': [0.2]}), Container({'a': [2.7], 'b': [2.4]}))])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_lamb_update(ws_n_grads_n_lr_n_wsnew, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    ws_raw, dcdws_raw, lr_raw, ws_raw_new = ws_n_grads_n_lr_n_wsnew
    ws = ws_raw.map(lambda x, _: ivy.variable(ivy.array(x)))
    dcdws = dcdws_raw.map(lambda x, _: ivy.array(x))
    lr = lr_raw.map(lambda x, _: ivy.array(x))
    ws_true_new = ws_raw_new.map(lambda x, _: ivy.variable(ivy.array(x)))
    mw = dcdws
    vw = dcdws.map(lambda x, _: x ** 2)
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
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.lamb_update)
