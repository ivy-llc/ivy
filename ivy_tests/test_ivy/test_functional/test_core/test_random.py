"""Collection of tests for unified reduction functions."""

# global
import pytest
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy
import ivy.functional.backends.numpy as ivy_np
import ivy_tests.test_ivy.helpers as helpers


# random_uniform
@given(
    data=st.data(),
    shape=helpers.get_shape(),
    dtype=st.sampled_from(ivy_np.valid_float_dtypes),
    tensor_fn=st.sampled_from([ivy.array]),
)
def test_random_uniform(data, shape, dtype, tensor_fn, device, call):
    bounds = data.draw(helpers.none_or_list_of_floats(dtype, 2))
    if bounds[0] is not None and bounds[1] is not None:
        low, high =  min(bounds), max(bounds)
    else:
        low, high = bounds[0], bounds[1]
    if low is None and high is not None:
        if high <= 0.0:
            return
    elif high is None and low is not None:
        if low >= 1.0:
            return
    ivy.seed(0)
    # smoke test
    if tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        return
    kwargs = {
        k: tensor_fn(v) for k, v in zip(["low", "high"], [low, high]) if v is not None
    }
    if shape is not None:
        kwargs["shape"] = shape
    ret = ivy.random_uniform(**kwargs, device=device)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    if shape is None:
        assert ret.shape == ()
    else:
        assert ret.shape == shape
    # value test
    ret_np = call(ivy.random_uniform, **kwargs, device=device)
    assert np.min((ret_np < (high if high else 1.0)).astype(np.int32)) == 1
    assert np.min((ret_np >= (low if low else 0.0)).astype(np.int32)) == 1


# random_normal
# @pytest.mark.parametrize("mean", [None, -1.0, 0.2])
# @pytest.mark.parametrize("std", [None, 0.5, 2.0])
# @pytest.mark.parametrize("shape", [None, (), (1, 2, 3)])
# @pytest.mark.parametrize("dtype", ["float32"])
# @pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn, lambda x: x])
@given(
    data=st.data(),
    shape=helpers.get_shape(),
    dtype=st.sampled_from(ivy_np.valid_float_dtypes),
    tensor_fn=st.sampled_from([ivy.array]),
)
def test_random_normal(data, shape, dtype, tensor_fn, device, call):
    params = data.draw(helpers.none_or_list_of_floats(dtype, 2))
    mean, std = params[0], params[1]
    ivy.seed(0)
    print('mean', mean)
    print('std', std)
    # smoke test
    if tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        return
    kwargs = {
        k: tensor_fn(v) for k, v in zip(["mean", "std"], [mean, std]) if v is not None
    }
    if shape is not None:
        kwargs["shape"] = shape
    ret = ivy.random_normal(**kwargs, device=device)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    if shape is None:
        assert ret.shape == ()
    else:
        assert ret.shape == shape
    # value test
    ret_np = call(ivy.random_normal, **kwargs, device=device)
    print('REACHED HERE')
    assert (
        np.min(
            (ret_np > (ivy.default(mean, 0.0) - 3 * ivy.default(std, 1.0))).astype(
                np.int32
            )
        )
        == 1
    )
    assert (
        np.min(
            (ret_np < (ivy.default(mean, 0.0) + 3 * ivy.default(std, 1.0))).astype(
                np.int32
            )
        )
        == 1
    )


# multinomial
@pytest.mark.parametrize("probs", [[[1.0, 2.0]], [[1.0, 0.5], [0.2, 0.3]], None])
@pytest.mark.parametrize("num_samples", [1, 2])
@pytest.mark.parametrize("replace", [True, False])
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_multinomial(probs, num_samples, replace, dtype, tensor_fn, device, call):
    population_size = 2
    if (
        call in [helpers.mx_call, helpers.tf_call, helpers.tf_graph_call]
        and not replace
    ):
        # mxnet and tenosorflow do not support multinomial without replacement
        pytest.skip()
    # smoke test
    probs = tensor_fn(probs, dtype, device) if probs is not None else probs
    batch_size = probs.shape[0] if probs is not None else 2
    ret = ivy.multinomial(population_size, num_samples, batch_size, probs, replace)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == tuple([batch_size] + [num_samples])


# randint
@pytest.mark.parametrize("low", [-1, 2])
@pytest.mark.parametrize("high", [5, 10])
@pytest.mark.parametrize("shape", [(), (1, 2, 3)])
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn, lambda x: x])
def test_randint(low, high, shape, dtype, tensor_fn, device, call):
    # smoke test
    if call in [helpers.mx_call, helpers.torch_call] and tensor_fn is helpers.var_fn:
        # PyTorch and MXNet do not support non-float variables
        pytest.skip()
    low_tnsr, high_tnsr = tensor_fn(low), tensor_fn(high)
    ret = ivy.randint(low_tnsr, high_tnsr, shape, device=device)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == shape
    # value test
    ret_np = call(ivy.randint, low_tnsr, high_tnsr, shape, device=device)
    assert np.min((ret_np < high).astype(np.int32)) == 1
    assert np.min((ret_np >= low).astype(np.int32)) == 1


# seed
@pytest.mark.parametrize("seed_val", [1, 2, 0])
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_seed(seed_val, dtype, tensor_fn, device, call):
    # smoke test
    ivy.seed(seed_val)


# shuffle
@pytest.mark.parametrize("x", [[1, 2, 3], [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]])
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_shuffle(x, dtype, tensor_fn, device, call):
    # smoke test
    x = tensor_fn(x, dtype, device)
    ret = ivy.shuffle(x)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    ivy.seed(0)
    first_shuffle = call(ivy.shuffle, x)
    ivy.seed(0)
    second_shuffle = call(ivy.shuffle, x)
    assert np.array_equal(first_shuffle, second_shuffle)
