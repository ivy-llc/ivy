"""Collection of tests for unified reduction functions."""

# global
import numpy as np
from hypothesis import given, assume, strategies as st

# local
import ivy
import ivy.functional.backends.numpy as ivy_np
import ivy_tests.test_ivy.helpers as helpers


# random_uniform
@given(
    data=st.data(),
    shape=helpers.get_shape(),
    dtype=st.sampled_from(ivy_np.valid_float_dtypes),
    as_variable=st.booleans(),
)
def test_random_uniform(data, shape, dtype, as_variable, device, call):
    low, high = data.draw(helpers.get_bounds(dtype=dtype))

    # mxnet does not support 0-dimensional variables
    assume(not (as_variable and call is helpers.mx_call))

    low_tnsr = ivy.array(low, dtype=dtype, device=device) if low is not None else None
    high_tnsr = (
        ivy.array(high, dtype=dtype, device=device) if high is not None else None
    )
    if as_variable and (low is not None):
        low_tnsr = ivy.variable(low_tnsr)
    if as_variable and (high is not None):
        high_tnsr = ivy.variable(high_tnsr)
    kwargs = {
        k: v for k, v in zip(["low", "high"], [low_tnsr, high_tnsr]) if v is not None
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
    assert (
        np.min((ret_np <= (high + abs(high) * 0.01 if high else 1.01)).astype(np.int32))
        == 1
    )
    assert (
        np.min((ret_np >= (low - abs(low) * 0.01 if low else -0.01)).astype(np.int32))
        == 1
    )


# random_normal
@given(
    data=st.data(),
    dtype=st.sampled_from(ivy_np.valid_float_dtypes),
    as_variable=st.booleans(),
)
def test_random_normal(data, dtype, as_variable, device, call):
    mean, std = data.draw(helpers.get_mean_std(dtype=dtype))
    ivy.seed(0)
    # smoke test
    assume(not (as_variable and call is helpers.mx_call))

    mean_tnsr = (
        ivy.array(mean, dtype=dtype, device=device) if mean is not None else None
    )
    std_tnsr = ivy.array(std, dtype=dtype, device=device) if std is not None else None
    if as_variable and (mean is not None):
        mean_tnsr = ivy.variable(mean_tnsr)
    if as_variable and (std is not None):
        std_tnsr = ivy.variable(std_tnsr)
    kwargs = {
        k: v for k, v in zip(["mean", "std"], [mean_tnsr, std_tnsr]) if v is not None
    }
    ret = ivy.random_normal(**kwargs, device=device)
    # type test
    assert ivy.is_ivy_array(ret)


@st.composite
def _pop_size_num_samples_replace_n_probs(draw):
    prob_dtype = draw(st.sampled_from(ivy_np.valid_float_dtypes))
    batch_size = draw(st.integers(1, 5))
    population_size = draw(st.integers(1, 20))
    replace = draw(st.booleans())
    if replace:
        num_samples = draw(st.integers(1, 20))
    else:
        num_samples = draw(st.integers(1, population_size))
    probs = draw(
        helpers.array_values(
            dtype=prob_dtype,
            shape=[batch_size, num_samples],
            min_value=0.0,
            max_value=1.0,
        )
        | st.just(None)
    )
    return prob_dtype, batch_size, population_size, num_samples, replace, probs


# multinomial
@given(everything=_pop_size_num_samples_replace_n_probs())
def test_multinomial(everything, device, call):
    prob_dtype, batch_size, population_size, num_samples, replace, probs = everything

    # tensorflow does not support multinomial without replacement
    assume(not (call is helpers.tf_call and not replace or prob_dtype == "float64"))

    # smoke test
    probs = (
        ivy.array(probs, dtype=prob_dtype, device=device)
        if probs is not None
        else probs
    )
    ret = ivy.multinomial(population_size, num_samples, batch_size, probs, replace)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == tuple([batch_size, num_samples])


# randint
@given(
    data=st.data(),
    shape=helpers.get_shape(allow_none=False),
    as_variable=st.booleans(),
)
def test_randint(data, shape, as_variable, device, call):
    dtype = ivy.default_int_dtype()
    # smoke test
    low, high = data.draw(helpers.get_bounds(dtype=dtype))

    # PyTorch and MXNet do not support non-float variables
    assume(
        not (
            call in [helpers.mx_call, helpers.torch_call]
            and as_variable
            or dtype == "uint64"
            or call == helpers.torch_call
            and dtype[0] == "u"
        )
    )

    low_tnsr = ivy.array(low, dtype=dtype, device=device)
    high_tnsr = ivy.array(high, dtype=dtype, device=device)
    if as_variable:
        low_tnsr, high_tnsr = ivy.variable(low_tnsr), ivy.variable(high_tnsr)
    kwargs = {
        k: v for k, v in zip(["low", "high"], [low_tnsr, high_tnsr]) if v is not None
    }
    kwargs["shape"] = shape
    ret = ivy.randint(**kwargs, device=device)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == shape
    # value test
    ret_np = call(ivy.randint, **kwargs, device=device)
    assert np.min((ret_np < high).astype(np.int64)) == 1
    assert np.min((ret_np >= low).astype(np.int64)) == 1


# seed
@given(
    seed_val=st.integers(min_value=0, max_value=2147483647),
)
def test_seed(seed_val):
    # smoke test
    ivy.seed(seed_val)


# shuffle
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_float_dtypes, min_num_dims=1
    ),
    as_variable=st.booleans(),
)
def test_shuffle(dtype_and_x, as_variable, device, call):
    # smoke test
    dtype, x = dtype_and_x
    x = ivy.array(x, dtype=dtype, device=device)
    if as_variable:
        x = ivy.variable(x)
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
