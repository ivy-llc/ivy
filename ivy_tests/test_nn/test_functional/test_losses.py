
# global
import pytest
import numpy as np

# local
import ivy
import ivy.numpy
import ivy_tests.helpers as helpers


# cross_entropy
@pytest.mark.parametrize(
    "t_n_p_n_res", [([[0., 1., 0.]], [[0.3, 0.2, 0.5]], [1.609438])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_cross_entropy(t_n_p_n_res, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    true, pred, true_target = t_n_p_n_res
    pred = tensor_fn(pred, dtype_str, dev_str)
    true = tensor_fn(true, dtype_str, dev_str)
    ret = ivy.cross_entropy(true, pred)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert list(ret.shape) == [1]
    # value test
    assert np.allclose(call(ivy.cross_entropy, true, pred), np.asarray(true_target))
    # compilation test
    if call in [helpers.torch_call]:
        # cross_entropy does not have backend implementation,
        # pytorch scripting requires direct bindings to work, which bypass get_framework()
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.cross_entropy)


# binary_cross_entropy
@pytest.mark.parametrize(
    "t_n_p_n_res", [([[0., 1., 0.]], [[0.3, 0.7, 0.5]], [[0.35667494, 0.35667494, 0.69314718]])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_binary_cross_entropy(t_n_p_n_res, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    true, pred, true_target = t_n_p_n_res
    pred = tensor_fn(pred, dtype_str, dev_str)
    true = tensor_fn(true, dtype_str, dev_str)
    ret = ivy.binary_cross_entropy(true, pred)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == pred.shape
    # value test
    assert np.allclose(call(ivy.binary_cross_entropy, true, pred), np.asarray(true_target))
    # compilation test
    if call in [helpers.torch_call]:
        # binary_cross_entropy does not have backend implementation,
        # pytorch scripting requires direct bindings to work, which bypass get_framework()
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.binary_cross_entropy)


# sparse_cross_entropy
@pytest.mark.parametrize(
    "t_n_p_n_res", [([1], [[0.3, 0.2, 0.5]], [1.609438])])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_sparse_cross_entropy(t_n_p_n_res, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    true, pred, true_target = t_n_p_n_res
    pred = tensor_fn(pred, dtype_str, dev_str)
    true = ivy.array(true, 'int32', dev_str)
    ret = ivy.sparse_cross_entropy(true, pred)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert list(ret.shape) == [1]
    # value test
    assert np.allclose(call(ivy.sparse_cross_entropy, true, pred), np.asarray(true_target))
    # compilation test
    if call in [helpers.torch_call]:
        # sparse_cross_entropy does not have backend implementation,
        # pytorch scripting requires direct bindings to work, which bypass get_framework()
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.sparse_cross_entropy)
