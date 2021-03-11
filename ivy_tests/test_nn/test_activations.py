
# global
import numpy as _np

# local
import ivy
import ivy_tests.helpers as helpers


def test_relu(dev_str, call):
    assert _np.allclose(call(ivy.relu, ivy.array([[-1., 1., 2.]])),
                        _np.array([[0., 1., 2.]]), atol=1e-6)
    helpers.assert_compilable(ivy.relu)


def test_leaky_relu(dev_str, call):
    assert _np.allclose(call(ivy.leaky_relu, ivy.array([[-1., 1., 2.]])),
                        _np.array([[-0.2, 1., 2.]]), atol=1e-6)
    helpers.assert_compilable(ivy.leaky_relu)


def test_tanh(dev_str, call):
    assert _np.allclose(call(ivy.tanh, ivy.array([[-1., 1., 2.]])),
                        _np.array([[-0.76159416,  0.76159416,  0.96402758]]), atol=1e-6)
    helpers.assert_compilable(ivy.tanh)


def test_sigmoid(dev_str, call):
    assert _np.allclose(call(ivy.sigmoid, ivy.array([[-1., 1., 2.]])),
                        _np.array([[0.26894142, 0.73105858, 0.88079708]]), atol=1e-6)
    helpers.assert_compilable(ivy.sigmoid)


def test_softmax(dev_str, call):
    assert _np.allclose(call(ivy.softmax, ivy.array([[1., 2., 3.]])),
                        _np.array([[0.09003057, 0.24472847, 0.66524096]]), atol=1e-6)
    helpers.assert_compilable(ivy.softmax)


def test_softplus(dev_str, call):
    assert _np.allclose(call(ivy.softplus, ivy.array([[1., 2., 3.]])),
                        _np.array([[1.31326169, 2.12692801, 3.04858735]]), atol=1e-6)
    helpers.assert_compilable(ivy.softplus)
