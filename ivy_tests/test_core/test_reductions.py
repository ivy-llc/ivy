"""
Collection of tests for templated reduction functions
"""

# global
import numpy as np

# local
import ivy
import ivy_tests.helpers as helpers


def test_reduce_sum():
    for lib, call in helpers.calls():
        assert np.array_equal(call(ivy.reduce_sum, ivy.array([1., 2., 3.], f=lib), 0, True),
                              np.sum(np.array([1., 2., 3.]), keepdims=True))
        assert np.array_equal(call(ivy.reduce_sum, ivy.array([1., 2., 3.], f=lib), (0,), True),
                              np.sum(np.array([1., 2., 3.]), keepdims=True))
        assert np.array_equal(call(ivy.reduce_sum, ivy.array([[1., 2., 3.]], f=lib), (0, 1), True),
                              np.sum(np.array([[1., 2., 3.]]), keepdims=True))
        helpers.assert_compilable('reduce_sum', lib)


def test_reduce_prod():
    for lib, call in helpers.calls():
        assert np.array_equal(call(ivy.reduce_prod, ivy.array([1., 2., 3.], f=lib), 0, True),
                              np.prod(np.array([1., 2., 3.]), keepdims=True))
        assert np.array_equal(call(ivy.reduce_prod, ivy.array([1., 2., 3.], f=lib), (0,), True),
                              np.prod(np.array([1., 2., 3.]), keepdims=True))
        assert np.array_equal(call(ivy.reduce_prod, ivy.array([[1., 2., 3.]], f=lib), (0, 1), True),
                              np.prod(np.array([[1., 2., 3.]]), keepdims=True))
        helpers.assert_compilable('reduce_prod', lib)


def test_reduce_mean():
    for lib, call in helpers.calls():
        assert np.array_equal(call(ivy.reduce_mean, ivy.array([1., 2., 3.], f=lib), 0, True),
                              np.mean(np.array([1., 2., 3.]), keepdims=True))
        assert np.array_equal(call(ivy.reduce_mean, ivy.array([1., 2., 3.], f=lib), (0,), True),
                              np.mean(np.array([1., 2., 3.]), keepdims=True))
        assert np.array_equal(call(ivy.reduce_mean, ivy.array([[1., 2., 3.]], f=lib), (0, 1), True),
                              np.mean(np.array([[1., 2., 3.]]), keepdims=True))
        helpers.assert_compilable('reduce_mean', lib)


def test_reduce_min():
    for lib, call in helpers.calls():
        assert np.array_equal(call(ivy.reduce_min, ivy.array([1., 2., 3.], f=lib),
                                   0, num_x_dims=1, keepdims=True),
                              np.min(np.array([1., 2., 3.]), keepdims=True))
        assert np.array_equal(call(ivy.reduce_min, ivy.array([1., 2., 3.], f=lib),
                                   num_x_dims=1, keepdims=True),
                              np.min(np.array([1., 2., 3.]), keepdims=True))
        assert np.array_equal(call(ivy.reduce_min, ivy.array([[1., 2., 3.]], f=lib),
                                   num_x_dims=2, keepdims=True),
                              np.min(np.array([[1., 2., 3.]]), keepdims=True))
        helpers.assert_compilable('reduce_min', lib)


def test_reduce_max():
    for lib, call in helpers.calls():
        assert np.array_equal(call(ivy.reduce_max, ivy.array([1., 2., 3.], f=lib),
                                   0, num_x_dims=1, keepdims=True),
                              np.max(np.array([1., 2., 3.]), keepdims=True))
        assert np.array_equal(call(ivy.reduce_max, ivy.array([1., 2., 3.], f=lib),
                                   num_x_dims=1, keepdims=True),
                              np.max(np.array([1., 2., 3.]), keepdims=True))
        assert np.array_equal(call(ivy.reduce_max, ivy.array([[1., 2., 3.]], f=lib),
                                   num_x_dims=2, keepdims=True),
                              np.max(np.array([[1., 2., 3.]]), keepdims=True))
        helpers.assert_compilable('reduce_max', lib)
