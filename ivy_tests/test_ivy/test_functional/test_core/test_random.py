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
    low, high = data.draw(helpers.get_bounds(dtype))
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
@given(
    data=st.data(),
    shape=helpers.get_shape(),
    dtype=st.sampled_from(ivy_np.valid_float_dtypes),
    tensor_fn=st.sampled_from([ivy.array]),
)
def test_random_normal(data, shape, dtype, tensor_fn, device, call):
    mean, std = data.draw(helpers.get_mean_std(dtype))
    ivy.seed(0)
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


# multinomial
@given(
    data=st.data(),
    num_samples=st.integers(1, 2),
    replace=st.booleans(),
    dtype=st.sampled_from(ivy_np.valid_float_dtypes),
    tensor_fn=st.sampled_from([ivy.array]),
)
def test_multinomial(data, num_samples, replace, dtype, tensor_fn, device, call):
    if dtype == 'float64':
        return
    probs, population_size = data.draw(helpers.get_probs(dtype))
    if (
        call in [helpers.mx_call, helpers.tf_call, helpers.tf_graph_call]
        and not replace
    ):
        # mxnet and tenosorflow do not support multinomial without replacement
        return
    # smoke test
    probs = tensor_fn(probs, dtype=dtype, device=device) if probs is not None else probs
    batch_size = probs.shape[0] if probs is not None else 2
    ret = ivy.multinomial(population_size, num_samples, batch_size, probs, replace)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == tuple([batch_size] + [num_samples])


# randint
# @pytest.mark.parametrize("low", [-1, 2])
# @pytest.mark.parametrize("high", [5, 10])
# @pytest.mark.parametrize("shape", [(), (1, 2, 3)])
# @pytest.mark.parametrize("dtype", ["float32"])
# @pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn, lambda x: x])
@given(
    data=st.data(),
    shape=helpers.get_shape(allow_none=False),
    dtype=st.sampled_from(ivy_np.valid_numeric_dtypes),
    as_variable=st.booleans()
)
def test_randint(data, shape, dtype, as_variable, device, call):
    # smoke test
    low, high = data.draw(helpers.get_bounds())
    if call in [helpers.mx_call, helpers.torch_call] and as_variable:
        # PyTorch and MXNet do not support non-float variables
        return

    low_tnsr = ivy.asarray(low, dtype=dtype, device=device)
    high_tnsr = ivy.asarray(high, dtype=dtype, device=device)
    if as_variable:
        low_tnsr, high_tnsr = ivy.variable(low), ivy.variable(high)
    kwargs = {
        k: v for k, v in zip(["low", "high"], [low_tnsr, high_tnsr]) if v is not None
    }
    kwargs['shape'] = shape

    ret = ivy.randint(**kwargs, device=device)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == shape
    # value test
    ret_np = call(ivy.randint, **kwargs, device=device)
    assert np.min((ret_np < high).astype(np.int32)) == 1
    assert np.min((ret_np >= low).astype(np.int32)) == 1


# seed
@given(seed_val=st.integers(min_value=0, max_value=2147483647))
def test_seed(seed_val):
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
