
# global
import pytest
import numpy as np

# local
import ivy
import ivy.functional.backends.numpy
import ivy_tests.test_ivy.helpers as helpers


# cross_entropy
@pytest.mark.parametrize(
    "t_n_p_n_res", [([[0., 1., 0.]], [[0.3, 0.2, 0.5]], [1.609438])])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
@pytest.mark.parametrize(
    "with_out", [False, True])
def test_cross_entropy(t_n_p_n_res, dtype, tensor_fn, with_out, dev, call):
    # smoke test
    true, pred, true_target = t_n_p_n_res
    pred = tensor_fn(pred, dtype, dev)
    true = tensor_fn(true, dtype, dev)

    # create dummy out
    out = ivy.zeros(1) if with_out else None

    ret = ivy.cross_entropy(true, pred, out=out)

    # type test
    assert ivy.is_native_array(ret)
    # cardinality test
    assert list(ret.shape) == [1]
    # value test
    assert np.allclose(ivy.to_numpy(ret), np.asarray(true_target))

    if with_out:
        assert np.allclose(ivy.to_numpy(ret), ivy.to_numpy(out))
    
        # check if native arrays are the same
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        
        # native array must be the same object 
        assert ret.data is out.data
    


# binary_cross_entropy
@pytest.mark.parametrize(
    "t_n_p_n_res", [([[0., 1., 0.]], [[0.3, 0.7, 0.5]], [[0.35667494, 0.35667494, 0.69314718]])])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
@pytest.mark.parametrize(
    "with_out", [False, True])
def test_binary_cross_entropy(t_n_p_n_res, dtype, tensor_fn, with_out, dev, call):
    # smoke test
    true, pred, true_target = t_n_p_n_res
    pred = tensor_fn(pred, dtype, dev)
    true = tensor_fn(true, dtype, dev)

    # create dummy out
    out = ivy.zeros(np.asarray(true_target).shape) if with_out else None

    ret = ivy.binary_cross_entropy(true, pred, out=out)

    # type test
    assert ivy.is_native_array(ret)
    # cardinality test
    assert ret.shape == pred.shape
    # value test
    assert np.allclose(ivy.to_numpy(ret), np.asarray(true_target))

    if with_out:
        assert np.allclose(ivy.to_numpy(ret), ivy.to_numpy(out))
    
        # check if native arrays are the same
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        
        # native array must be the same object 
        assert ret.data is out.data


# sparse_cross_entropy
@pytest.mark.parametrize(
    "t_n_p_n_res", [([1], [[0.3, 0.2, 0.5]], [1.609438])])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
@pytest.mark.parametrize(
    "with_out", [False, True])
def test_sparse_cross_entropy(t_n_p_n_res, dtype, tensor_fn, with_out, dev, call):
    # smoke test
    true, pred, true_target = t_n_p_n_res
    pred = tensor_fn(pred, dtype, dev)
    true = ivy.array(true, 'int32', dev)

    # create dummy out
    out = ivy.zeros(np.asarray(true_target).shape) if with_out else None

    ret = ivy.sparse_cross_entropy(true, pred, out=out)
    # type test
    assert ivy.is_native_array(ret)
    # cardinality test
    assert list(ret.shape) == [1]

    # value test
    assert np.allclose(ivy.to_numpy(ret), np.asarray(true_target))

    if with_out:
        assert np.allclose(ivy.to_numpy(ret), ivy.to_numpy(out))
    
        # check if native arrays are the same
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        
        # native array must be the same object 
        assert ret.data is out.data
