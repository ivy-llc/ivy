"""
Collection of tests for templated logic functions
"""

# global
import numpy as np

# local
import ivy
import ivy_tests.helpers as helpers


def test_logical_and():
    for f, call in helpers.f_n_calls():
        assert np.array_equal(call(ivy.logical_and, ivy.array([True, True], f=f),
                                   ivy.array([False, True], f=f), f=f),
                              np.logical_and(np.array([True, True]), np.array([False, True])))
        assert np.array_equal(call(ivy.logical_and, ivy.array([[0.]], f=f),
                                   ivy.array([[1.]], f=f), f=f),
                              np.logical_and(np.array([[0.]]), np.array([[1.]])))
        if call in [helpers.torch_call]:
            # pytorch scripting does not support .type() method
            continue
        helpers.assert_compilable('logical_and', f)


def test_logical_or():
    for f, call in helpers.f_n_calls():
        assert np.array_equal(call(ivy.logical_or, ivy.array([True, True], f=f),
                                   ivy.array([False, True], f=f), f=f),
                              np.logical_or(np.array([True, True]), np.array([False, True])))
        assert np.array_equal(call(ivy.logical_or, ivy.array([[0.]], f=f),
                                   ivy.array([[1.]], f=f), f=f),
                              np.logical_or(np.array([[0.]]), np.array([[1.]])))
        if call in [helpers.torch_call]:
            # pytorch scripting does not support .type() method
            continue
        helpers.assert_compilable('logical_or', f)


def test_logical_not():
    for f, call in helpers.f_n_calls():
        assert np.array_equal(call(ivy.logical_not, ivy.array([True, True], f=f)),
                              np.logical_not(np.array([True, True])))
        assert np.array_equal(call(ivy.logical_not, ivy.array([[0.]], f=f)),
                              np.logical_not(np.array([[0.]])))
        if call in [helpers.torch_call]:
            # pytorch scripting does not support .type() method
          continue
        helpers.assert_compilable('logical_not', f)
