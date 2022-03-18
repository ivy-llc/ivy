"""
Collection of tests for unified linear algebra functions
"""

# global
import pytest
import numpy as np

# local
import ivy
import ivy.functional.backends.numpy
import ivy_tests.test_ivy.helpers as helpers


# vector_to_skew_symmetric_matrix
@pytest.mark.parametrize(
    "x", [[[[1., 2., 3.]], [[4., 5., 6.]], [[1., 2., 3.]], [[4., 5., 6.]], [[1., 2., 3.]]], [[1., 2., 3.]]])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_vector_to_skew_symmetric_matrix(x, dtype, tensor_fn, dev, call):
    # smoke test
    x = tensor_fn(x, dtype, dev)
    ret = ivy.vector_to_skew_symmetric_matrix(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape + (x.shape[-1],)
    # value test
    assert np.allclose(call(ivy.vector_to_skew_symmetric_matrix, x),
                       ivy.functional.backends.numpy.vector_to_skew_symmetric_matrix(ivy.to_numpy(x)))
