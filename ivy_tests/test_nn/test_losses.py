
# global
import pytest
import numpy as np

# local
import ivy
import ivy.numpy
import ivy_tests.helpers as helpers


# binary_cross_entropy
@pytest.mark.parametrize(
    "x_n_y_n_res", [([[0.3, 0.7, 0.5]], [[0., 1., 0.]], [[0.35667494, 0.35667494, 0.69314718]])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_binary_cross_entropy(x_n_y_n_res, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x, y, true_target = x_n_y_n_res
    x = tensor_fn(x, dtype_str, dev_str)
    y = tensor_fn(y, dtype_str, dev_str)
    ret = ivy.binary_cross_entropy(x, y)
    # type test
    assert isinstance(ret, ivy.Array)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    assert np.allclose(call(ivy.binary_cross_entropy, x, y), np.asarray(true_target))
    # compilation test
    if call in [helpers.torch_call]:
        # binary_cross_entropy does not have backend implementation,
        # pytorch scripting requires direct bindings to work, which bypass get_framework()
        return
    helpers.assert_compilable(ivy.binary_cross_entropy)
