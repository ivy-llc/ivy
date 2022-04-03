"""
Collection of tests for unified logic functions
"""

# global
import pytest
import numpy as np

# local
import ivy
import ivy.functional.backends.numpy
import ivy_tests.test_ivy.helpers as helpers



# logical_not
@pytest.mark.parametrize(
    "x", [[True, True], [[0.]]])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_logical_not(x, dtype, tensor_fn, dev, call):
    # smoke test
    x = tensor_fn(x, dtype, dev)
    ret = ivy.logical_not(x)
    # type test
    assert ivy.is_native_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.logical_not, x), ivy.functional.backends.numpy.logical_not(ivy.to_numpy(x)))
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support .type() method
        return
