"""
Collection of tests for templated gradient functions
"""

# global
import pytest
import numpy as np

# local
import ivy
import ivy_tests.helpers as helpers
from ivy.core.container import Container


def test_variable(dev_str, call):
    if call is helpers.tf_graph_call:
        # cannot create variables as part of compiled tf graph
        pytest.skip()
    call(ivy.variable, ivy.array([0.]))
    call(ivy.variable, ivy.array([0.], 'float32'))
    call(ivy.variable, ivy.array([[0.]]))
    if call in [helpers.torch_call]:
        # pytorch scripting does not support attribute setting
        return
    helpers.assert_compilable(ivy.variable)


def test_execute_with_gradients(dev_str, call):
    # func with single return val
    func = lambda xs_in: (xs_in['w'] * xs_in['w'])[0]
    xs = Container({'w': ivy.variable(ivy.array([3.]))})
    y, dydxs = call(ivy.execute_with_gradients, func, xs)
    assert np.allclose(y, np.array(9.))
    if call is helpers.np_call:
        # numpy doesn't support autodiff
        assert dydxs is None
    else:
        assert np.allclose(ivy.to_numpy(dydxs['w']), np.array([6.]))

    # func with multi return vals
    func = lambda xs_in: ((xs_in['w'] * xs_in['w'])[0], xs_in['w'] * 1.5)
    xs = Container({'w': ivy.variable(ivy.array([3.]))})
    y, dydxs, extra_out = call(ivy.execute_with_gradients, func, xs)
    assert np.allclose(y, np.array(9.))
    assert np.allclose(extra_out, np.array([4.5]))
    if call is helpers.np_call:
        # numpy doesn't support autodiff
        assert dydxs is None
    else:
        assert np.allclose(ivy.to_numpy(dydxs['w']), np.array([6.]))

    # func with multi weights vals
    func = lambda xs_in: (xs_in['w1'] * xs_in['w2'])[0]
    xs = Container({'w1': ivy.variable(ivy.array([3.])),
                    'w2': ivy.variable(ivy.array([5.]))})
    y, dydxs = call(ivy.execute_with_gradients, func, xs)
    assert np.allclose(y, np.array(15.))
    if call is helpers.np_call:
        # numpy doesn't support autodiff
        assert dydxs is None
    else:
        assert np.allclose(ivy.to_numpy(dydxs['w1']), np.array([5.]))
        assert np.allclose(ivy.to_numpy(dydxs['w2']), np.array([3.]))

    # compile
    if call in [helpers.torch_call]:
        # pytorch scripting does not support internal function definitions
        return
    helpers.assert_compilable(ivy.execute_with_gradients)


def test_gradient_descent_update(dev_str, call):
    ws = Container({'w': ivy.variable(ivy.array([3.]))})
    dcdws = Container({'w': ivy.array([6.])})
    w_new = ivy.array(ivy.gradient_descent_update(ws, dcdws, 0.1)['w'])
    assert np.allclose(ivy.to_numpy(w_new), np.array([2.4]))
    if call in [helpers.torch_call]:
        # pytorch scripting does not support internal function definitions
        return
    helpers.assert_compilable(ivy.gradient_descent_update)


def test_adam_update(dev_str, call):
    ws = Container({'w': ivy.variable(ivy.array([3.]))})
    dcdws = Container({'w': ivy.array([6.])})
    mw = dcdws
    vw = dcdws.map(lambda x, _: x ** 2)
    w_new = ivy.array(ivy.adam_update(ws, dcdws, 0.1, mw, vw, ivy.array(1))[0]['w'])
    assert np.allclose(ivy.to_numpy(w_new), np.array([2.96837726]))
    if call in [helpers.torch_call]:
        # pytorch scripting does not support internal function definitions
        return
    helpers.assert_compilable(ivy.adam_update)


def test_stop_gradient(dev_str, call):
    x_init = ivy.array([0.])
    x_init_np = call(lambda x: x, x_init)
    x_new = call(ivy.stop_gradient, x_init)
    assert np.array_equal(x_init_np, x_new)
    if call in [helpers.torch_call]:
        # pytorch scripting does not support attribute setting
        return
    helpers.assert_compilable(ivy.stop_gradient)
