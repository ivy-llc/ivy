"""Collection of tests for unified reduction functions."""

# global
from hypothesis import strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test


# random_uniform
@handle_test(
    fn_tree="functional.ivy.random_uniform",
    dtype_and_low=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-1000,
        max_value=100,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=2,
    ),
    dtype_and_high=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=101,
        max_value=1000,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=2,
    ),
    dtype=helpers.get_dtypes("float", full=False),
    seed=helpers.ints(min_value=0, max_value=100),
    test_gradients=st.just(False),
)
def test_random_uniform(
    *,
    dtype_and_low,
    dtype_and_high,
    dtype,
    seed,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    low_dtype, low = dtype_and_low
    high_dtype, high = dtype_and_high

    def call():
        return helpers.test_function(
            ground_truth_backend=ground_truth_backend,
            input_dtypes=low_dtype + high_dtype,
            test_flags=test_flags,
            on_device=on_device,
            fw=backend_fw,
            fn_name=fn_name,
            test_values=False,
            low=low[0],
            high=high[0],
            shape=None,
            dtype=dtype[0],
            seed=seed,
        )

    ret, ret_gt = call()
    if seed:
        ret1, ret_gt2 = call()
        assert ivy.any(ret == ret1)
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)

    for (u, v) in zip(ret, ret_gt):
        assert u.dtype == v.dtype


# random_normal
@handle_test(
    fn_tree="functional.ivy.random_normal",
    dtype_and_mean=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-1000,
        max_value=1000,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=2,
    ),
    dtype_and_std=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=1000,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=2,
    ),
    dtype=helpers.get_dtypes("float", full=False),
    seed=helpers.ints(min_value=0, max_value=100),
    test_gradients=st.just(False),
)
def test_random_normal(
    dtype_and_mean,
    dtype_and_std,
    dtype,
    seed,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    mean_dtype, mean = dtype_and_mean
    std_dtype, std = dtype_and_std

    def call():
        return helpers.test_function(
            ground_truth_backend=ground_truth_backend,
            input_dtypes=mean_dtype + std_dtype,
            test_flags=test_flags,
            on_device=on_device,
            fw=backend_fw,
            fn_name=fn_name,
            test_values=False,
            mean=mean[0],
            std=std[0],
            shape=None,
            dtype=dtype[0],
            seed=seed,
        )

    ret, ret_gt = call()
    if seed:
        ret1, ret_gt1 = call()
        assert ivy.any(ret == ret1)
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    for (u, v) in zip(ret, ret_gt):
        assert u.dtype == v.dtype


@st.composite
def _pop_size_num_samples_replace_n_probs(draw):
    prob_dtype = draw(helpers.get_dtypes("float", full=False))
    batch_size = draw(helpers.ints(min_value=1, max_value=5))
    population_size = draw(helpers.ints(min_value=1, max_value=20))
    replace = draw(st.booleans())
    if replace:
        num_samples = draw(helpers.ints(min_value=1, max_value=20))
    else:
        num_samples = draw(helpers.ints(min_value=1, max_value=population_size))
    probs = draw(
        helpers.array_values(
            dtype=prob_dtype[0],
            shape=[batch_size, num_samples],
            min_value=1.0013580322265625e-05,
            max_value=1.0,
            exclude_min=True,
            large_abs_safety_factor=2,
            safety_factor_scale="linear",
        )
    )
    return prob_dtype, batch_size, population_size, num_samples, replace, probs


# multinomial
@handle_test(
    fn_tree="functional.ivy.multinomial",
    everything=_pop_size_num_samples_replace_n_probs(),
    seed=helpers.ints(min_value=0, max_value=100),
    test_gradients=st.just(False),
)
def test_multinomial(
    *,
    everything,
    seed,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    prob_dtype, batch_size, population_size, num_samples, replace, probs = everything

    def call():
        return helpers.test_function(
            ground_truth_backend=ground_truth_backend,
            input_dtypes=prob_dtype,
            test_flags=test_flags,
            on_device=on_device,
            fw=backend_fw,
            fn_name=fn_name,
            test_values=False,
            population_size=population_size,
            num_samples=num_samples,
            batch_size=batch_size,
            probs=probs[0] if probs is not None else probs,
            replace=replace,
            seed=seed,
        )

    ret = call()

    if not ivy.exists(ret):
        return

    ret_np, ret_from_np = ret
    if seed:
        ret_np1, ret_from_np1 = call()

        assert ivy.any(ret_np == ret_np1)

    ret_np = helpers.flatten_and_to_np(ret=ret_np)
    ret_from_np = helpers.flatten_and_to_np(ret=ret_from_np)
    for (u, v) in zip(ret_np, ret_from_np):
        assert u.dtype == v.dtype
        assert u.shape == v.shape


@st.composite
def _gen_randint_data(draw):
    dtype = draw(helpers.get_dtypes("signed_integer", full=False))
    shape1, shape2, shape3 = draw(helpers.mutually_broadcastable_shapes(num_shapes=3))
    shape, shape_low, shape_high = sorted([shape1, shape2, shape3])
    shape = draw(st.sampled_from([None, shape]))
    low = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=shape_low,
            min_value=-100,
            max_value=25,
        )
    )
    high = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=shape_high,
            min_value=26,
            max_value=100,
        )
    )
    return dtype, low, high, shape


# randint
@handle_test(
    fn_tree="functional.ivy.randint",
    dtype_low_high_shape=_gen_randint_data(),
    seed=helpers.ints(min_value=0, max_value=100),
    test_gradients=st.just(False),
)
def test_randint(
    *,
    dtype_low_high_shape,
    seed,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, low, high, shape = dtype_low_high_shape

    def call():
        return helpers.test_function(
            ground_truth_backend=ground_truth_backend,
            input_dtypes=dtype,
            test_flags=test_flags,
            on_device=on_device,
            fw=backend_fw,
            fn_name=fn_name,
            test_values=False,
            return_flat_np_arrays=True,
            low=low,
            high=high,
            shape=shape,
            dtype=dtype[0],
            seed=seed,
        )

    ret, ret_gt = call()
    if seed:
        ret1, ret_gt1 = call()
        assert ivy.all(ret[0] == ret1[0])
    def _get_indices(shape1, shape2):
        indices = [0] * len(shape1)
        for i, (p, q) in enumerate(zip(shape1[::-1], shape2[::-1])):
            if p == q:
                indices[i] = None
        return tuple(indices[::-1])

    low = ivy.array(low)
    high = ivy.array(high)

    if shape is not None:
        low, high, _ = ivy.broadcast_arrays(low, high, ivy.ones(shape))
    else:
        low, high = ivy.broadcast_arrays(low, high)

    if shape and not high.shape == tuple(shape):
        indices = _get_indices(high.shape, shape)
        high = high[indices]
        low = low[indices]
    low = helpers.flatten_and_to_np(ret=low)
    high = helpers.flatten_and_to_np(ret=high)
    assert ivy.all(ret[0] >= low[0]) and ivy.all(ret[0] < high[0])
    assert ivy.all(ret_gt[0] >= low[0]) and ivy.all(ret_gt[0] < high[0])


# seed
@handle_test(
    fn_tree="functional.ivy.seed",
    seed_val=helpers.ints(min_value=0, max_value=2147483647),
)
def test_seed(seed_val):
    # smoke test
    ivy.seed(seed_value=seed_val)


# shuffle
@handle_test(
    fn_tree="functional.ivy.shuffle",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
        min_num_dims=1,
        min_dim_size=2,
    ),
    seed=helpers.ints(min_value=0, max_value=100),
    test_gradients=st.just(False),
)
def test_shuffle(
    *,
    dtype_and_x,
    seed,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, x = dtype_and_x

    def call():
        return helpers.test_function(
            ground_truth_backend=ground_truth_backend,
            input_dtypes=dtype,
            test_flags=test_flags,
            on_device=on_device,
            fw=backend_fw,
            fn_name=fn_name,
            test_values=False,
            x=x[0],
            seed=seed,
        )

    ret, ret_gt = call()
    if seed:
        ret1, ret_gt1 = call()
        assert ivy.any(ret == ret1)
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    for (u, v) in zip(ret, ret_gt):
        assert ivy.all(ivy.sort(u, axis=0) == ivy.sort(v, axis=0))
