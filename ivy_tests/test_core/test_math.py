"""
Collection of tests for templated math functions
"""

# global
import numpy as np

# local
import ivy
import ivy_tests.helpers as helpers


def test_sin():
    for f, call in helpers.f_n_calls():
        assert np.allclose(call(ivy.sin, ivy.array([0.], f=f)), np.sin(np.array([0.])))
        assert np.allclose(call(ivy.sin, ivy.array([[0.]], f=f)), np.sin(np.array([[0.]])))
        helpers.assert_compilable('sin', f)


def test_cos():
    for f, call in helpers.f_n_calls():
        assert np.allclose(call(ivy.cos, ivy.array([0.], f=f)), np.cos(np.array([0.])))
        assert np.allclose(call(ivy.cos, ivy.array([[0.]], f=f)), np.cos(np.array([[0.]])))
        helpers.assert_compilable('cos', f)


def test_tan():
    for f, call in helpers.f_n_calls():
        assert np.allclose(call(ivy.tan, ivy.array([0.], f=f)), np.tan(np.array([0.])))
        assert np.allclose(call(ivy.tan, ivy.array([[0.]], f=f)), np.tan(np.array([[0.]])))
        helpers.assert_compilable('tan', f)


def test_asin():
    for f, call in helpers.f_n_calls():
        assert np.allclose(call(ivy.asin, ivy.array([0.], f=f)), np.arcsin(np.array([0.])))
        assert np.allclose(call(ivy.asin, ivy.array([[0.]], f=f)), np.arcsin(np.array([[0.]])))
        helpers.assert_compilable('asin', f)


def test_acos():
    for f, call in helpers.f_n_calls():
        assert np.allclose(call(ivy.acos, ivy.array([0.], f=f)), np.arccos(np.array([0.])))
        assert np.allclose(call(ivy.acos, ivy.array([[0.]], f=f)), np.arccos(np.array([[0.]])))
        helpers.assert_compilable('acos', f)


def test_atan():
    for f, call in helpers.f_n_calls():
        assert np.allclose(call(ivy.atan, ivy.array([0.], f=f)), np.arctan(np.array([0.])))
        assert np.allclose(call(ivy.atan, ivy.array([[0.]], f=f)), np.arctan(np.array([[0.]])))
        helpers.assert_compilable('atan', f)


def test_atan2():
    for f, call in helpers.f_n_calls():
        if call is helpers.mx_graph_call:
            # mxnet symbolic does not support atan2
            continue
        assert np.array_equal(call(ivy.atan2, ivy.array([0.], f=f), ivy.array([0.], f=f)),
                              np.arctan2(np.array([0.]), np.array([0.])))
        assert np.array_equal(call(ivy.atan2, ivy.array([[0.]], f=f), ivy.array([[0.]], f=f)),
                              np.arctan2(np.array([[0.]]), np.array([[0.]])))
        helpers.assert_compilable('atan2', f)


def test_sinh():
    for f, call in helpers.f_n_calls():
        assert np.allclose(call(ivy.sinh, ivy.array([[.1, .2, .3]], f=f), f=f),
                           np.sinh(np.array([[.1, .2, .3]])))  # almost equal
        helpers.assert_compilable('sinh', f)


def test_cosh():
    for f, call in helpers.f_n_calls():
        assert np.allclose(call(ivy.cosh, ivy.array([[.1, .2, .3]], f=f), f=f),
                           np.cosh(np.array([[.1, .2, .3]])))  # almost equal
        helpers.assert_compilable('cosh', f)


def test_tanh():
    for f, call in helpers.f_n_calls():
        assert np.allclose(call(ivy.tanh, ivy.array([[.1, .2, .3]], f=f), f=f),
                           np.tanh(np.array([[.1, .2, .3]])))  # almost equal
        helpers.assert_compilable('tanh', f)


def test_asinh():
    for f, call in helpers.f_n_calls():
        assert np.allclose(call(ivy.asinh, ivy.array([[.1, .2, .3]], f=f), f=f),
                           np.arcsinh(np.array([[.1, .2, .3]])))  # almost equal
        helpers.assert_compilable('asinh', f)


def test_acosh():
    for f, call in helpers.f_n_calls():
        assert np.allclose(call(ivy.acosh, ivy.array([[1, 1.2, 200, 10000]], f=f), f=f),
                           np.arccosh(np.array([[1, 1.2, 200, 10000]])))  # almost equal
        helpers.assert_compilable('acosh', f)


def test_atanh():
    for f, call in helpers.f_n_calls():
        assert np.allclose(call(ivy.atanh, ivy.array([[-0.5, 0.5]], f=f), f=f),
                           np.arctanh(np.array([[-0.5, 0.5]])))  # almost equal
        helpers.assert_compilable('atanh', f)


def test_log():
    for f, call in helpers.f_n_calls():
        assert np.array_equal(call(ivy.log, ivy.array([1.], f=f)), np.array([0.]))
        assert np.array_equal(call(ivy.log, ivy.array([[1.]], f=f)), np.array([[0.]]))
        helpers.assert_compilable('log', f)


def test_exp():
    for f, call in helpers.f_n_calls():
        assert np.array_equal(call(ivy.exp, ivy.array([0.], f=f)), np.array([1.]))
        assert np.array_equal(call(ivy.exp, ivy.array([[0.]], f=f)), np.array([[1.]]))
        helpers.assert_compilable('exp', f)
