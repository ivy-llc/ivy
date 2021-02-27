"""
Collection of tests for templated math functions
"""

# global
import numpy as np

# local
import ivy
import ivy_tests.helpers as helpers


def test_sin():
    for lib, call in helpers.calls():
        assert np.allclose(call(ivy.sin, ivy.array([0.], f=lib)), np.sin(np.array([0.])))
        assert np.allclose(call(ivy.sin, ivy.array([[0.]], f=lib)), np.sin(np.array([[0.]])))
        helpers.assert_compilable('sin', lib)


def test_cos():
    for lib, call in helpers.calls():
        assert np.allclose(call(ivy.cos, ivy.array([0.], f=lib)), np.cos(np.array([0.])))
        assert np.allclose(call(ivy.cos, ivy.array([[0.]], f=lib)), np.cos(np.array([[0.]])))
        helpers.assert_compilable('cos', lib)


def test_tan():
    for lib, call in helpers.calls():
        assert np.allclose(call(ivy.tan, ivy.array([0.], f=lib)), np.tan(np.array([0.])))
        assert np.allclose(call(ivy.tan, ivy.array([[0.]], f=lib)), np.tan(np.array([[0.]])))
        helpers.assert_compilable('tan', lib)


def test_asin():
    for lib, call in helpers.calls():
        assert np.allclose(call(ivy.asin, ivy.array([0.], f=lib)), np.arcsin(np.array([0.])))
        assert np.allclose(call(ivy.asin, ivy.array([[0.]], f=lib)), np.arcsin(np.array([[0.]])))
        helpers.assert_compilable('asin', lib)


def test_acos():
    for lib, call in helpers.calls():
        assert np.allclose(call(ivy.acos, ivy.array([0.], f=lib)), np.arccos(np.array([0.])))
        assert np.allclose(call(ivy.acos, ivy.array([[0.]], f=lib)), np.arccos(np.array([[0.]])))
        helpers.assert_compilable('acos', lib)


def test_atan():
    for lib, call in helpers.calls():
        assert np.allclose(call(ivy.atan, ivy.array([0.], f=lib)), np.arctan(np.array([0.])))
        assert np.allclose(call(ivy.atan, ivy.array([[0.]], f=lib)), np.arctan(np.array([[0.]])))
        helpers.assert_compilable('atan', lib)


def test_atan2():
    for lib, call in helpers.calls():
        if call is helpers.mx_graph_call:
            # mxnet symbolic does not support atan2
            continue
        assert np.array_equal(call(ivy.atan2, ivy.array([0.], f=lib), ivy.array([0.], f=lib)),
                              np.arctan2(np.array([0.]), np.array([0.])))
        assert np.array_equal(call(ivy.atan2, ivy.array([[0.]], f=lib), ivy.array([[0.]], f=lib)),
                              np.arctan2(np.array([[0.]]), np.array([[0.]])))
        helpers.assert_compilable('atan2', lib)


def test_sinh():
    for lib, call in helpers.calls():
        assert np.allclose(call(ivy.sinh, ivy.array([[.1, .2, .3]], f=lib), f=lib),
                           np.sinh(np.array([[.1, .2, .3]])))  # almost equal
        helpers.assert_compilable('sinh', lib)


def test_cosh():
    for lib, call in helpers.calls():
        assert np.allclose(call(ivy.cosh, ivy.array([[.1, .2, .3]], f=lib), f=lib),
                           np.cosh(np.array([[.1, .2, .3]])))  # almost equal
        helpers.assert_compilable('cosh', lib)


def test_tanh():
    for lib, call in helpers.calls():
        assert np.allclose(call(ivy.tanh, ivy.array([[.1, .2, .3]], f=lib), f=lib),
                           np.tanh(np.array([[.1, .2, .3]])))  # almost equal
        helpers.assert_compilable('tanh', lib)


def test_asinh():
    for lib, call in helpers.calls():
        assert np.allclose(call(ivy.asinh, ivy.array([[.1, .2, .3]], f=lib), f=lib),
                           np.arcsinh(np.array([[.1, .2, .3]])))  # almost equal
        helpers.assert_compilable('asinh', lib)


def test_acosh():
    for lib, call in helpers.calls():
        assert np.allclose(call(ivy.acosh, ivy.array([[1, 1.2, 200, 10000]], f=lib), f=lib),
                           np.arccosh(np.array([[1, 1.2, 200, 10000]])))  # almost equal
        helpers.assert_compilable('acosh', lib)


def test_atanh():
    for lib, call in helpers.calls():
        assert np.allclose(call(ivy.atanh, ivy.array([[-0.5, 0.5]], f=lib), f=lib),
                           np.arctanh(np.array([[-0.5, 0.5]])))  # almost equal
        helpers.assert_compilable('atanh', lib)


def test_log():
    for lib, call in helpers.calls():
        assert np.array_equal(call(ivy.log, ivy.array([1.], f=lib)), np.array([0.]))
        assert np.array_equal(call(ivy.log, ivy.array([[1.]], f=lib)), np.array([[0.]]))
        helpers.assert_compilable('log', lib)


def test_exp():
    for lib, call in helpers.calls():
        assert np.array_equal(call(ivy.exp, ivy.array([0.], f=lib)), np.array([1.]))
        assert np.array_equal(call(ivy.exp, ivy.array([[0.]], f=lib)), np.array([[1.]]))
        helpers.assert_compilable('exp', lib)
