# global
from hypothesis import strategies as st, given
import jax

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.frontends.jax as jax_frontend
from ivy_tests.test_ivy.helpers import handle_frontend_test

"""
Tests for jax.random cannot be made normally since a `uint32` PRNG key must
be passed for jax, meaning torch and paddle would always fail.
The format used below should be followed for jax.random tests
"""


@st.composite
def _get_minval_maxval(draw):
    interval = draw(st.integers(min_value=1, max_value=50))
    minval = draw(st.floats(min_value=-100, max_value=100))
    maxval = minval + interval
    return minval, maxval


@given(
    key=st.integers(min_value=int(-1e15), max_value=int(1e15)),
    shape=helpers.get_shape(),
    dtype=helpers.get_dtypes("float", full=False, prune_function=False),
    minval_maxval=_get_minval_maxval(),
)
def test_jax_uniform(
    key,
    shape,
    dtype,
    minval_maxval,
):
    minval, maxval = minval_maxval

    jax_frontend.config.update("jax_enable_x64", True)
    frontend_prng_key = jax_frontend.random.PRNGKey(key)
    frontend_ret = jax_frontend.random.uniform(
        frontend_prng_key, shape, dtype[0], minval, maxval
    )

    framework_prng_key = jax.random.PRNGKey(key)
    framework_ret = jax.random.uniform(
        framework_prng_key, shape, dtype[0], minval, maxval
    )

    assert frontend_ret.ivy_array.dtype == framework_ret.dtype.name
    assert framework_ret.shape == framework_ret.shape


@given(
    key=st.integers(min_value=int(-1e15), max_value=int(1e15)),
    shape=helpers.get_shape(),
    dtype=helpers.get_dtypes("float", full=False, prune_function=False),
)
def test_jax_normal(
    key,
    shape,
    dtype,
):
    jax_frontend.config.update("jax_enable_x64", True)
    frontend_prng_key = jax_frontend.random.PRNGKey(key)
    frontend_ret = jax_frontend.random.normal(frontend_prng_key, shape, dtype[0])

    framework_prng_key = jax.random.PRNGKey(key)
    framework_ret = jax.random.normal(framework_prng_key, shape, dtype[0])

    assert frontend_ret.ivy_array.dtype == framework_ret.dtype.name
    assert framework_ret.shape == framework_ret.shape


@given(
    key=st.integers(min_value=int(-1e15), max_value=int(1e15)),
    alpha=st.floats(min_value=0, max_value=5, exclude_min=True),
    beta=st.floats(min_value=0, max_value=5, exclude_min=True),
    shape=helpers.get_shape(
        min_num_dims=2, max_num_dims=2, min_dim_size=1, max_dim_size=5
    ),
    dtype=helpers.get_dtypes("float", full=False, prune_function=False),
)
def test_jax_beta(key, alpha, beta, shape, dtype):

    jax_frontend.config.update("jax_enable_x64", True)
    frontend_prng_key = jax_frontend.random.PRNGKey(key)
    frontend_ret = jax_frontend.random.beta(
        frontend_prng_key, alpha, beta, shape, dtype[0]
    )

    framework_prng_key = jax.random.PRNGKey(key)
    framework_ret = jax.random.beta(framework_prng_key, alpha, beta, shape, dtype[0])

    assert frontend_ret.ivy_array.dtype == framework_ret.dtype.name
    assert framework_ret.shape == framework_ret.shape


@handle_frontend_test(
    fn_tree="jax.random.dirichlet",
    dtype_key=helpers.dtype_and_values(
        available_dtypes=["uint32"],
        min_value=0,
        max_value=2000,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
        max_dim_size=2,
    ),
    dtype_alpha=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", full=False),
        shape=st.tuples(
            st.integers(min_value=2, max_value=5),
        ),
        min_value=1.1,
        max_value=100.0,
        exclude_min=True,
    ),
    shape=helpers.get_shape(
        min_num_dims=2, max_num_dims=2, min_dim_size=2, max_dim_size=5
    ),
    dtype=helpers.get_dtypes("float", full=False),
    test_with_out=st.just(False),
)
def test_jax_dirichlet(
    *,
    dtype_key,
    dtype_alpha,
    shape,
    dtype,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, key = dtype_key
    _, alpha = dtype_alpha

    def call():
        return helpers.test_frontend_function(
            input_dtypes=input_dtype,
            frontend=frontend,
            test_flags=test_flags,
            fn_tree=fn_tree,
            on_device=on_device,
            test_values=False,
            key=key[0],
            alpha=alpha[0],
            shape=shape,
            dtype=dtype[0],
        )

    ret = call()

    if not ivy.exists(ret):
        return

    ret_np, ret_from_np = ret
    ret_np = helpers.flatten_and_to_np(ret=ret_np)
    ret_from_np = helpers.flatten_and_to_np(ret=ret_from_np)
    for (u, v) in zip(ret_np, ret_from_np):
        assert u.dtype == v.dtype
        assert u.shape == v.shape
