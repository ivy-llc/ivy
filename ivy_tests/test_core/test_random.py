"""
Collection of tests for templated reduction functions
"""

# global
import pytest
import numpy as np

# local
import ivy
import ivy.numpy
import ivy_tests.helpers as helpers


# random_uniform
@pytest.mark.parametrize(
    "low", [None, -1., 0.2])
@pytest.mark.parametrize(
    "high", [None, 0.5, 2.])
@pytest.mark.parametrize(
    "shape", [None, (), (1, 2, 3)])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn, lambda x: x])
def test_random_uniform(low, high, shape, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    if tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    kwargs = dict([(k, tensor_fn(v)) for k, v in zip(['low', 'high'], [low, high]) if v is not None])
    if shape is not None:
        kwargs['shape'] = shape
    ret = ivy.random_uniform(**kwargs, dev_str=dev_str)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    if shape is None:
        assert ret.shape == ()
    else:
        assert ret.shape == shape
    # value test
    ret_np = call(ivy.random_uniform, **kwargs, dev_str=dev_str)
    assert np.min((ret_np < (high if high else 1.)).astype(np.int32)) == 1
    assert np.min((ret_np > (low if low else 0.)).astype(np.int32)) == 1
    # compilation test
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.random_uniform)


# random_normal
@pytest.mark.parametrize(
    "mean", [None, -1., 0.2])
@pytest.mark.parametrize(
    "std", [None, 0.5, 2.])
@pytest.mark.parametrize(
    "shape", [None, (), (1, 2, 3)])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn, lambda x: x])
def test_random_normal(mean, std, shape, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    ivy.seed(0)
    if tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()
    kwargs = dict([(k, tensor_fn(v)) for k, v in zip(['mean', 'std'], [mean, std]) if v is not None])
    if shape is not None:
        kwargs['shape'] = shape
    ret = ivy.random_normal(**kwargs, dev_str=dev_str)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    if shape is None:
        assert ret.shape == ()
    else:
        assert ret.shape == shape
    # value test
    ret_np = call(ivy.random_normal, **kwargs, dev_str=dev_str)
    assert np.min((ret_np > (ivy.default(mean, 0.) - 3*ivy.default(std, 1.))).astype(np.int32)) == 1
    assert np.min((ret_np < (ivy.default(mean, 0.) + 3*ivy.default(std, 1.))).astype(np.int32)) == 1
    # compilation test
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.random_normal)


# multinomial
@pytest.mark.parametrize(
    "probs", [[[1., 2.]], [[1., 0.5], [0.2, 0.3]], None])
@pytest.mark.parametrize(
    "num_samples", [1, 2])
@pytest.mark.parametrize(
    "replace", [True, False])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_multinomial(probs, num_samples, replace, dtype_str, tensor_fn, dev_str, call):
    population_size = 2
    if call in [helpers.mx_call, helpers.tf_call, helpers.tf_graph_call] and not replace:
        # mxnet and tenosorflow do not support multinomial without replacement
        pytest.skip()
    # smoke test
    probs = tensor_fn(probs, dtype_str, dev_str) if probs is not None else probs
    batch_size = probs.shape[0] if probs is not None else 2
    ret = ivy.multinomial(population_size, num_samples, batch_size, probs, replace)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == tuple([batch_size] + [num_samples])
    # compilation test
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.multinomial)


# randint
@pytest.mark.parametrize(
    "low", [-1, 2])
@pytest.mark.parametrize(
    "high", [5, 10])
@pytest.mark.parametrize(
    "shape", [(), (1, 2, 3)])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn, lambda x: x])
def test_randint(low, high, shape, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    if call in [helpers.mx_call, helpers.torch_call] and tensor_fn is helpers.var_fn:
        # PyTorch and MXNet do not support non-float variables
        pytest.skip()
    low_tnsr, high_tnsr = tensor_fn(low), tensor_fn(high)
    ret = ivy.randint(low_tnsr, high_tnsr, shape, dev_str=dev_str)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == shape
    # value test
    ret_np = call(ivy.randint, low_tnsr, high_tnsr, shape, dev_str=dev_str)
    assert np.min((ret_np < high).astype(np.int32)) == 1
    assert np.min((ret_np >= low).astype(np.int32)) == 1
    # compilation test
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.randint)


# seed
@pytest.mark.parametrize(
    "seed_val", [1, 2, 0])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_seed(seed_val, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    ivy.seed(seed_val)
    # compilation test
    if call in [helpers.torch_call]:
        # pytorch scripting does not support functions with None return
        return
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.seed)


# shuffle
@pytest.mark.parametrize(
    "x", [[1, 2, 3], [[1., 4.], [2., 5.], [3., 6.]]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_shuffle(x, dtype_str, tensor_fn, dev_str, call):
    # smoke test
    x = tensor_fn(x, dtype_str, dev_str)
    ret = ivy.shuffle(x)
    # type test
    assert ivy.is_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    ivy.seed(0)
    first_shuffle = call(ivy.shuffle, x)
    ivy.seed(0)
    second_shuffle = call(ivy.shuffle, x)
    assert np.array_equal(first_shuffle, second_shuffle)
    # compilation test
    if not ivy.wrapped_mode():
        helpers.assert_compilable(ivy.shuffle)
