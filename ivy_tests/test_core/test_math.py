"""
Collection of tests for templated math functions
"""

# global
import numpy as np

# local
import ivy
import ivy_tests.helpers as helpers


def test_sin(dev_str, call):
    assert np.allclose(call(ivy.sin, ivy.tensor([0.])), np.sin(np.array([0.])))
    assert np.allclose(call(ivy.sin, ivy.tensor([[0.]])), np.sin(np.array([[0.]])))
    helpers.assert_compilable(ivy.sin)


def test_cos(dev_str, call):
    assert np.allclose(call(ivy.cos, ivy.tensor([0.])), np.cos(np.array([0.])))
    assert np.allclose(call(ivy.cos, ivy.tensor([[0.]])), np.cos(np.array([[0.]])))
    helpers.assert_compilable(ivy.cos)


def test_tan(dev_str, call):
    assert np.allclose(call(ivy.tan, ivy.tensor([0.])), np.tan(np.array([0.])))
    assert np.allclose(call(ivy.tan, ivy.tensor([[0.]])), np.tan(np.array([[0.]])))
    helpers.assert_compilable(ivy.tan)


def test_asin(dev_str, call):
    assert np.allclose(call(ivy.asin, ivy.tensor([0.])), np.arcsin(np.array([0.])))
    assert np.allclose(call(ivy.asin, ivy.tensor([[0.]])), np.arcsin(np.array([[0.]])))
    helpers.assert_compilable(ivy.asin)


def test_acos(dev_str, call):
    assert np.allclose(call(ivy.acos, ivy.tensor([0.])), np.arccos(np.array([0.])))
    assert np.allclose(call(ivy.acos, ivy.tensor([[0.]])), np.arccos(np.array([[0.]])))
    helpers.assert_compilable(ivy.acos)


def test_atan(dev_str, call):
    assert np.allclose(call(ivy.atan, ivy.tensor([0.])), np.arctan(np.array([0.])))
    assert np.allclose(call(ivy.atan, ivy.tensor([[0.]])), np.arctan(np.array([[0.]])))
    helpers.assert_compilable(ivy.atan)


def test_atan2(dev_str, call):
    assert np.array_equal(call(ivy.atan2, ivy.tensor([0.]), ivy.tensor([0.])),
                          np.arctan2(np.array([0.]), np.array([0.])))
    assert np.array_equal(call(ivy.atan2, ivy.tensor([[0.]]), ivy.tensor([[0.]])),
                          np.arctan2(np.array([[0.]]), np.array([[0.]])))
    helpers.assert_compilable(ivy.atan2)


def test_sinh(dev_str, call):
    assert np.allclose(call(ivy.sinh, ivy.tensor([[.1, .2, .3]])),
                       np.sinh(np.array([[.1, .2, .3]])))  # almost equal
    helpers.assert_compilable(ivy.sinh)


def test_cosh(dev_str, call):
    assert np.allclose(call(ivy.cosh, ivy.tensor([[.1, .2, .3]])),
                       np.cosh(np.array([[.1, .2, .3]])))  # almost equal
    helpers.assert_compilable(ivy.cosh)


def test_tanh(dev_str, call):
    assert np.allclose(call(ivy.tanh, ivy.tensor([[.1, .2, .3]])),
                       np.tanh(np.array([[.1, .2, .3]])))  # almost equal
    helpers.assert_compilable(ivy.tanh)


def test_asinh(dev_str, call):
    assert np.allclose(call(ivy.asinh, ivy.tensor([[.1, .2, .3]])),
                       np.arcsinh(np.array([[.1, .2, .3]])))  # almost equal
    helpers.assert_compilable(ivy.asinh)


def test_acosh(dev_str, call):
    assert np.allclose(call(ivy.acosh, ivy.tensor([[1, 1.2, 200, 10000]])),
                       np.arccosh(np.array([[1, 1.2, 200, 10000]])))  # almost equal
    helpers.assert_compilable(ivy.acosh)


def test_atanh(dev_str, call):
    assert np.allclose(call(ivy.atanh, ivy.tensor([[-0.5, 0.5]])),
                       np.arctanh(np.array([[-0.5, 0.5]])))  # almost equal
    helpers.assert_compilable(ivy.atanh)


def test_log(dev_str, call):
    assert np.array_equal(call(ivy.log, ivy.tensor([1.])), np.array([0.]))
    assert np.array_equal(call(ivy.log, ivy.tensor([[1.]])), np.array([[0.]]))
    helpers.assert_compilable(ivy.log)


def test_exp(dev_str, call):
    assert np.array_equal(call(ivy.exp, ivy.tensor([0.])), np.array([1.]))
    assert np.array_equal(call(ivy.exp, ivy.tensor([[0.]])), np.array([[1.]]))
    helpers.assert_compilable(ivy.exp)
