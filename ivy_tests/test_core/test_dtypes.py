"""
Collection of tests for unified general functions
"""

# global
import pytest

# local
import ivy
import ivy.backends.numpy
import ivy.backends.jax
import ivy.backends.tensorflow
import ivy.backends.torch
import ivy.backends.mxnet
import ivy_tests.helpers as helpers


# dtype objects
def test_dtype_instances(dev_str, call):
    assert ivy.exists(ivy.int8)
    assert ivy.exists(ivy.int16)
    assert ivy.exists(ivy.int32)
    assert ivy.exists(ivy.int64)
    assert ivy.exists(ivy.uint8)
    assert ivy.exists(ivy.uint16)
    assert ivy.exists(ivy.uint32)
    assert ivy.exists(ivy.uint64)
    assert ivy.exists(ivy.float32)
    assert ivy.exists(ivy.float64)
    assert ivy.exists(ivy.bool)


# cast
@pytest.mark.parametrize(
    "object_in", [[1], [True], [[1., 2.]]])
@pytest.mark.parametrize(
    "starting_dtype_str", ['float32', 'int32', 'bool'])
@pytest.mark.parametrize(
    "target_dtype_str", ['float32', 'int32', 'bool'])
def test_cast(object_in, starting_dtype_str, target_dtype_str, dev_str, call):
    # smoke test
    x = ivy.array(object_in, starting_dtype_str, dev_str)
    ret = ivy.cast(x, target_dtype_str)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert ivy.dtype_str(ret) == target_dtype_str
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support .type() method
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.cast)
