"""
Collection of tests for templated logic functions
"""

# global
import numpy as np

# local
import ivy
import ivy_tests.helpers as helpers


def test_logical_and():
    for lib, call in helpers.calls():
        assert np.array_equal(call(ivy.logical_and, ivy.array([True, True], f=lib),
                                   ivy.array([False, True], f=lib), f=lib),
                              np.logical_and(np.array([True, True]), np.array([False, True])))
        assert np.array_equal(call(ivy.logical_and, ivy.array([[0.]], f=lib),
                                   ivy.array([[1.]], f=lib), f=lib),
                              np.logical_and(np.array([[0.]]), np.array([[1.]])))
        if call in [helpers.torch_call]:
            # pytorch scripting does not support .type() method
            continue
        helpers.assert_compilable('logical_and', lib)


def test_logical_or():
    for lib, call in helpers.calls():
        assert np.array_equal(call(ivy.logical_or, ivy.array([True, True], f=lib),
                                   ivy.array([False, True], f=lib), f=lib),
                              np.logical_or(np.array([True, True]), np.array([False, True])))
        assert np.array_equal(call(ivy.logical_or, ivy.array([[0.]], f=lib),
                                   ivy.array([[1.]], f=lib), f=lib),
                              np.logical_or(np.array([[0.]]), np.array([[1.]])))
        if call in [helpers.torch_call]:
            # pytorch scripting does not support .type() method
            continue
        helpers.assert_compilable('logical_or', lib)


def test_logical_not():
    for lib, call in helpers.calls():
        assert np.array_equal(call(ivy.logical_not, ivy.array([True, True], f=lib)),
                              np.logical_not(np.array([True, True])))
        assert np.array_equal(call(ivy.logical_not, ivy.array([[0.]], f=lib)),
                              np.logical_not(np.array([[0.]])))
        if call in [helpers.torch_call]:
            # pytorch scripting does not support .type() method
          continue
        helpers.assert_compilable('logical_not', lib)
