"""
PyTorch Pointwise Ops Frontend Unit Tests
"""

# global
import pytest
import numpy as np
from numbers import Number

# local
import ivy
import ivy_tests.helpers as helpers
import ivy.frontends.torch as ivy_torch

# abs
@pytest.mark.parametrize(
    "x_n_x_absed", [(-2.5, 2.5), ([-10.7], [10.7]), ([[-3.8, 2.2], [1.7, -0.2]], [[3.8, 2.2], [1.7, 0.2]])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_abs(x_n_x_absed, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    if (isinstance(x_n_x_absed[0], Number) or isinstance(x_n_x_absed[1], Number))\
            and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    x = tensor_fn(x_n_x_absed[0], dtype_str, dev_str)
    ret = ivy_torch.abs(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.abs, x), np.array(x_n_x_absed[1]))
    # compilation test
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.abs)
