"""
Collection of tests for templated math functions
"""

# global
import numpy as np

# local
import ivy.core.math as ivy_math
import ivy.core.general as ivy_gen
import ivy_tests.helpers as helpers


def test_sin():
    for lib, call in helpers.calls:
        assert np.allclose(call(ivy_math.sin, ivy_gen.array([0.], f=lib)), np.sin(np.array([0.])))
        assert np.allclose(call(ivy_math.sin, ivy_gen.array([[0.]], f=lib)), np.sin(np.array([[0.]])))


def test_cos():
    for lib, call in helpers.calls:
        assert np.allclose(call(ivy_math.cos, ivy_gen.array([0.], f=lib)), np.cos(np.array([0.])))
        assert np.allclose(call(ivy_math.cos, ivy_gen.array([[0.]], f=lib)), np.cos(np.array([[0.]])))


def test_tan():
    for lib, call in helpers.calls:
        assert np.allclose(call(ivy_math.tan, ivy_gen.array([0.], f=lib)), np.tan(np.array([0.])))
        assert np.allclose(call(ivy_math.tan, ivy_gen.array([[0.]], f=lib)), np.tan(np.array([[0.]])))


def test_asin():
    for lib, call in helpers.calls:
        assert np.allclose(call(ivy_math.asin, ivy_gen.array([0.], f=lib)), np.arcsin(np.array([0.])))
        assert np.allclose(call(ivy_math.asin, ivy_gen.array([[0.]], f=lib)), np.arcsin(np.array([[0.]])))


def test_acos():
    for lib, call in helpers.calls:
        assert np.allclose(call(ivy_math.acos, ivy_gen.array([0.], f=lib)), np.arccos(np.array([0.])))
        assert np.allclose(call(ivy_math.acos, ivy_gen.array([[0.]], f=lib)), np.arccos(np.array([[0.]])))


def test_atan():
    for lib, call in helpers.calls:
        assert np.allclose(call(ivy_math.atan, ivy_gen.array([0.], f=lib)), np.arctan(np.array([0.])))
        assert np.allclose(call(ivy_math.atan, ivy_gen.array([[0.]], f=lib)), np.arctan(np.array([[0.]])))


def test_atan2():
    for lib, call in helpers.calls:
        if call is helpers.mx_graph_call:
            # mxnet symbolic does not support atan2
            continue
        assert np.array_equal(call(ivy_math.atan2, ivy_gen.array([0.], f=lib), ivy_gen.array([0.], f=lib)),
                              np.arctan2(np.array([0.]), np.array([0.])))
        assert np.array_equal(call(ivy_math.atan2, ivy_gen.array([[0.]], f=lib), ivy_gen.array([[0.]], f=lib)),
                              np.arctan2(np.array([[0.]]), np.array([[0.]])))


def test_sinh():
    for lib, call in helpers.calls:
        assert np.allclose(call(ivy_math.sinh, ivy_gen.array([[.1, .2, .3]], f=lib), f=lib),
                           np.sinh(np.array([[.1, .2, .3]])))  # almost equal


def test_cosh():
    for lib, call in helpers.calls:
        assert np.allclose(call(ivy_math.cosh, ivy_gen.array([[.1, .2, .3]], f=lib), f=lib),
                           np.cosh(np.array([[.1, .2, .3]])))  # almost equal


def test_tanh():
    for lib, call in helpers.calls:
        assert np.allclose(call(ivy_math.tanh, ivy_gen.array([[.1, .2, .3]], f=lib), f=lib),
                           np.tanh(np.array([[.1, .2, .3]])))  # almost equal


def test_asinh():
    for lib, call in helpers.calls:
        assert np.allclose(call(ivy_math.asinh, ivy_gen.array([[.1, .2, .3]], f=lib), f=lib),
                           np.arcsinh(np.array([[.1, .2, .3]])))  # almost equal


def test_acosh():
    for lib, call in helpers.calls:
        assert np.allclose(call(ivy_math.acosh, ivy_gen.array([[1, 1.2, 200, 10000]], f=lib), f=lib),
                           np.arccosh(np.array([[1, 1.2, 200, 10000]])))  # almost equal


def test_atanh():
    for lib, call in helpers.calls:
        assert np.allclose(call(ivy_math.atanh, ivy_gen.array([[-0.5, 0.5]], f=lib), f=lib),
                           np.arctanh(np.array([[-0.5, 0.5]])))  # almost equal


def test_log():
    for lib, call in helpers.calls:
        assert np.array_equal(call(ivy_math.log, ivy_gen.array([1.], f=lib)), np.array([0.]))
        assert np.array_equal(call(ivy_math.log, ivy_gen.array([[1.]], f=lib)), np.array([[0.]]))


def test_exp():
    for lib, call in helpers.calls:
        assert np.array_equal(call(ivy_math.exp, ivy_gen.array([0.], f=lib)), np.array([1.]))
        assert np.array_equal(call(ivy_math.exp, ivy_gen.array([[0.]], f=lib)), np.array([[1.]]))
