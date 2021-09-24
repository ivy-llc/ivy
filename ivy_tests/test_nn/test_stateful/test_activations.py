"""
Collection of tests for templated neural network activations
"""

# global
import pytest
import numpy as np

# local
import ivy
import ivy_tests.helpers as helpers


# GEGLU
@pytest.mark.parametrize(
    "bs_oc_target", [
        ([1, 2], 10, [[[0., 0.02189754, 0.04893785, 0.08134944, 0.11933776,
                        0.16308454, 0.21274757, 0.26846102, 0.3303356, 0.39845937],
                       [0., 0.02189754, 0.04893785, 0.08134944, 0.11933776,
                        0.16308454, 0.21274757, 0.26846102, 0.3303356, 0.39845937]]])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_geglu(bs_oc_target, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    batch_shape, output_channels, target = bs_oc_target
    x = ivy.cast(ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), output_channels*2), 'float32')
    geglu_layer = ivy.GEGLU()
    ret = geglu_layer(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == tuple(batch_shape + [output_channels])
    # value test
    assert np.allclose(call(geglu_layer, x), np.array(target))
    # compilation test
    if call is helpers.torch_call:
        # pytest scripting does not **kwargs
        return
    if ivy.wrapped_mode():
        helpers.assert_compilable(geglu_layer)
