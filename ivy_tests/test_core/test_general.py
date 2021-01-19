"""
Collection of tests for templated general functions
"""

# global
import numpy as np

# local
import ivy_tests.helpers as helpers
import ivy.core.general as ivy_gen


def test_array():
    for lib, call in helpers.calls:
        assert np.array_equal(call(ivy_gen.array, [0.], f=lib), np.array([0.]))
        assert np.array_equal(call(ivy_gen.array, [0.], 'float32', f=lib), np.array([0.], dtype=np.float32))
        assert np.array_equal(call(ivy_gen.array, [[0.]], f=lib), np.array([[0.]]))
