"""
Collection of tests for templated logic functions
"""

# global
import numpy as np

# local
import ivy.core.general as ivy_gen
import ivy.core.logic as ivy_logic
import ivy_tests.helpers as helpers


def test_logical_and():
    for lib, call in helpers.calls:
        assert np.array_equal(call(ivy_logic.logical_and, ivy_gen.array([True, True], f=lib),
                                   ivy_gen.array([False, True], f=lib), f=lib),
                              np.logical_and(np.array([True, True]), np.array([False, True])))
        assert np.array_equal(call(ivy_logic.logical_and, ivy_gen.array([[0.]], f=lib),
                                   ivy_gen.array([[1.]], f=lib), f=lib),
                              np.logical_and(np.array([[0.]]), np.array([[1.]])))


def test_logical_or():
    for lib, call in helpers.calls:
        assert np.array_equal(call(ivy_logic.logical_or, ivy_gen.array([True, True], f=lib),
                                   ivy_gen.array([False, True], f=lib), f=lib),
                              np.logical_or(np.array([True, True]), np.array([False, True])))
        assert np.array_equal(call(ivy_logic.logical_or, ivy_gen.array([[0.]], f=lib),
                                   ivy_gen.array([[1.]], f=lib), f=lib),
                              np.logical_or(np.array([[0.]]), np.array([[1.]])))


def test_logical_not():
    for lib, call in helpers.calls:
        assert np.array_equal(call(ivy_logic.logical_not, ivy_gen.array([True, True], f=lib)),
                              np.logical_not(np.array([True, True])))
        assert np.array_equal(call(ivy_logic.logical_not, ivy_gen.array([[0.]], f=lib)),
                              np.logical_not(np.array([[0.]])))
