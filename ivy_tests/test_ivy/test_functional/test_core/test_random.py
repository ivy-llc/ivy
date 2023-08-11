"""Collection of tests for unified reduction functions."""

# global
from hypothesis import strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test, update_backend


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
    on_device
):
    low_dtype, low = dtype_and_low
    high_dtype, high = dtype_and_high

    def call():
        return helpers.test_function(
            input_dtypes=low_dtype + high_dtype,
            test_flags=test_flags,
            on_device=on_device,
            backend_to_test=backend_fw,
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
    ret = helpers.flatten_and_to_np(ret=ret, backend=backend_fw)
    ret_gt = helpers.flatten_and_to_np(
        ret=ret_gt, backend=test_flags.ground_truth_backend
    )

    for u, v in zip(ret, ret_gt):
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
):
    mean_dtype, mean = dtype_and_mean
    std_dtype, std = dtype_and_std

    def call():
        return helpers.test_function(
            input_dtypes=mean_dtype + std_dtype,
            test_flags=test_flags,
            on_device=on_device,
            backend_to_test=backend_fw,
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
    ret = helpers.flatten_and_to_np(ret=ret, backend=backend_fw)
    ret_gt = helpers.flatten_and_to_np(
        ret=ret_gt, backend=test_flags.ground_truth_backend
    )
    for u, v in zip(ret, ret_gt):
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
    test_instance_method=st.just(False),
)
def test_multinomial(*, everything, seed, test_flags, backend_fw, fn_name, on_device):
    prob_dtype, batch_size, population_size, num_samples, replace, probs = everything

    def call():
        return helpers.test_function(
            input_dtypes=prob_dtype,
            test_flags=test_flags,
            on_device=on_device,
            backend_to_test=backend_fw,
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

    ret_np = helpers.flatten_and_to_np(ret=ret_np, backend=backend_fw)
    ret_from_np = helpers.flatten_and_to_np(
        ret=ret_from_np, backend=test_flags.ground_truth_backend
    )
    for u, v in zip(ret_np, ret_from_np):
        assert u.dtype == v.dtype
        assert u.shape == v.shape


@st.composite
def _gen_randint_data(draw):
    dtype = draw(helpers.get_dtypes("signed_integer", full=False))
    dim1 = draw(helpers.ints(min_value=1, max_value=5))
    dim2 = draw(helpers.ints(min_value=2, max_value=8))
    low = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=(dim1, dim2),
            min_value=-100,
            max_value=25,
        )
    )
    high = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=(dim1, dim2),
            min_value=26,
            max_value=100,
        )
    )
    return dtype, low, high


# randint
@handle_test(
    fn_tree="functional.ivy.randint",
    dtype_low_high=_gen_randint_data(),
    seed=helpers.ints(min_value=0, max_value=100),
    test_gradients=st.just(False),
)
def test_randint(*, dtype_low_high, seed, test_flags, backend_fw, fn_name, on_device):
    dtype, low, high = dtype_low_high

    def call():
        return helpers.test_function(
            input_dtypes=dtype,
            test_flags=test_flags,
            on_device=on_device,
            backend_to_test=backend_fw,
            fn_name=fn_name,
            test_values=False,
            low=low,
            high=high,
            shape=None,
            dtype=dtype[0],
            seed=seed,
        )

    ret, ret_gt = call()
    if seed:
        ret1, ret_gt1 = call()
        assert ivy.any(ret == ret1)
    ret = helpers.flatten_and_to_np(ret=ret, backend=backend_fw)
    ret_gt = helpers.flatten_and_to_np(
        ret=ret_gt, backend=test_flags.ground_truth_backend
    )
    for u, v in zip(ret, ret_gt):
        assert ivy.all(u >= low) and ivy.all(u < high)
        assert ivy.all(v >= low) and ivy.all(v < high)


# seed
@handle_test(
    fn_tree="functional.ivy.seed",
    seed_val=helpers.ints(min_value=0, max_value=2147483647),
)
def test_seed(seed_val, backend_fw):
    # smoke test
    with update_backend(backend_fw) as ivy_backend:
        ivy_backend.seed(seed_value=seed_val)


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
    axis=helpers.ints(min_value=0, max_value=0),
    test_gradients=st.just(False),
)
def test_shuffle(
    *, dtype_and_x, seed, axis, test_flags, backend_fw, fn_name, on_device
):
    dtype, x = dtype_and_x

    def call():
        return helpers.test_function(
            input_dtypes=dtype,
            test_flags=test_flags,
            on_device=on_device,
            backend_to_test=backend_fw,
            fn_name=fn_name,
            test_values=False,
            x=x[0],
            axis=axis,
            seed=seed,
        )

    ret, ret_gt = call()
    if seed:
        ret1, ret_gt1 = call()
        assert ivy.any(ret == ret1)
    ret = helpers.flatten_and_to_np(ret=ret, backend=backend_fw)
    ret_gt = helpers.flatten_and_to_np(
        ret=ret_gt, backend=test_flags.ground_truth_backend
    )
    for u, v in zip(ret, ret_gt):
        assert ivy.all(ivy.sort(u, axis=0) == ivy.sort(v, axis=0))
