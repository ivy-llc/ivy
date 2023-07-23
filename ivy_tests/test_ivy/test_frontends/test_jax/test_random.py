# global
import pytest
from hypothesis import strategies as st
import ivy

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# ToDo: Find solution around torch and paddle not running with uints 32,
#  64 and remove xfail fixture


@st.composite
def _get_minval_maxval(draw):
    interval = draw(st.integers(min_value=1, max_value=50))
    minval = draw(st.floats(min_value=-100, max_value=100))
    maxval = draw(st.floats(min_value=-100 + interval, max_value=100 + interval))
    return minval, maxval


@pytest.mark.xfail
@handle_frontend_test(
    fn_tree="jax.random.uniform",
    dtype_key=helpers.dtype_and_values(
        available_dtypes=["uint32"],
        min_value=0,
        max_value=2000,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
        max_dim_size=2,
    ),
    shape=helpers.get_shape(),
    dtype=helpers.get_dtypes("float", full=False),
    dtype_minval_maxval=_get_minval_maxval(),
)
def test_jax_uniform(
    *,
    dtype_key,
    shape,
    dtype,
    dtype_minval_maxval,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, key = dtype_key
    minval, maxval = dtype_minval_maxval

    def call():
        return helpers.test_frontend_function(
            input_dtypes=input_dtype,
            frontend=frontend,
            test_flags=test_flags,
            backend_to_test=backend_fw,
            fn_tree=fn_tree,
            on_device=on_device,
            test_values=False,
            key=key[0],
            shape=shape,
            dtype=dtype[0],
            minval=minval,
            maxval=maxval,
        )

    ret = call()

    if not ivy.exists(ret):
        return

    ret_np, ret_from_np = ret
    ret_np = helpers.flatten_and_to_np(ret=ret_np, backend=backend_fw)
    ret_from_np = helpers.flatten_and_to_np(ret=ret_from_np, backend=backend_fw)
    for u, v in zip(ret_np, ret_from_np):
        assert u.dtype == v.dtype
        assert u.shape == v.shape


@pytest.mark.xfail
@handle_frontend_test(
    fn_tree="jax.random.normal",
    dtype_key=helpers.dtype_and_values(
        available_dtypes=["uint32"],
        min_value=0,
        max_value=2000,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
        max_dim_size=2,
    ),
    shape=helpers.get_shape(),
    dtype=helpers.get_dtypes("float", full=False),
)
def test_jax_normal(
    *,
    dtype_key,
    shape,
    dtype,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, key = dtype_key

    def call():
        return helpers.test_frontend_function(
            input_dtypes=input_dtype,
            frontend=frontend,
            test_flags=test_flags,
            backend_to_test=backend_fw,
            fn_tree=fn_tree,
            on_device=on_device,
            test_values=False,
            key=key[0],
            shape=shape,
            dtype=dtype[0],
        )

    ret = call()

    if not ivy.exists(ret):
        return

    ret_np, ret_from_np = ret
    ret_np = helpers.flatten_and_to_np(ret=ret_np, backend=backend_fw)
    ret_from_np = helpers.flatten_and_to_np(ret=ret_from_np, backend=backend_fw)
    for u, v in zip(ret_np, ret_from_np):
        assert u.dtype == v.dtype
        assert u.shape == v.shape


@pytest.mark.xfail
@handle_frontend_test(
    fn_tree="jax.random.beta",
    dtype_key=helpers.dtype_and_values(
        available_dtypes=["uint32"],
        min_value=0,
        max_value=2000,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
        max_dim_size=2,
    ),
    alpha=st.floats(min_value=0, max_value=5, exclude_min=True),
    beta=st.floats(min_value=0, max_value=5, exclude_min=True),
    shape=helpers.get_shape(
        min_num_dims=2, max_num_dims=2, min_dim_size=1, max_dim_size=5
    ),
    dtype=helpers.get_dtypes("float", full=False),
    test_with_out=st.just(False),
)
def test_jax_beta(
    *,
    dtype_key,
    alpha,
    beta,
    shape,
    dtype,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, key = dtype_key

    def call():
        return helpers.test_frontend_function(
            input_dtypes=input_dtype,
            frontend=frontend,
            test_flags=test_flags,
            backend_to_test=backend_fw,
            fn_tree=fn_tree,
            on_device=on_device,
            test_values=False,
            key=key[0],
            a=alpha,
            b=beta,
            shape=shape,
            dtype=dtype[0],
        )

    ret = call()

    if not ivy.exists(ret):
        return

    ret_np, ret_from_np = ret
    ret_np = helpers.flatten_and_to_np(ret=ret_np, backend=backend_fw)
    ret_from_np = helpers.flatten_and_to_np(ret=ret_from_np, backend=backend_fw)
    for u, v in zip(ret_np, ret_from_np):
        assert u.dtype == v.dtype
        assert u.shape == v.shape


@pytest.mark.xfail
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
    backend_fw,
):
    input_dtype, key = dtype_key
    _, alpha = dtype_alpha

    def call():
        return helpers.test_frontend_function(
            input_dtypes=input_dtype,
            frontend=frontend,
            test_flags=test_flags,
            backend_to_test=backend_fw,
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
    ret_np = helpers.flatten_and_to_np(ret=ret_np, backend=backend_fw)
    ret_from_np = helpers.flatten_and_to_np(ret=ret_from_np, backend=backend_fw)
    for u, v in zip(ret_np, ret_from_np):
        assert u.dtype == v.dtype
        assert u.shape == v.shape


@pytest.mark.xfail
@handle_frontend_test(
    fn_tree="jax.random.cauchy",
    dtype_key=helpers.dtype_and_values(
        available_dtypes=["uint32"],
        min_value=0,
        max_value=2000,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
        max_dim_size=2,
    ),
    shape=helpers.get_shape(),
    dtype=helpers.get_dtypes("float", full=False),
)
def test_jax_cauchy(
    *,
    dtype_key,
    shape,
    dtype,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, key = dtype_key

    def call():
        return helpers.test_frontend_function(
            input_dtypes=input_dtype,
            frontend=frontend,
            test_flags=test_flags,
            backend_to_test=backend_fw,
            fn_tree=fn_tree,
            on_device=on_device,
            test_values=False,
            key=key[0],
            shape=shape,
            dtype=dtype[0],
        )

    ret = call()

    if not ivy.exists(ret):
        return

    ret_np, ret_from_np = ret
    ret_np = helpers.flatten_and_to_np(ret=ret_np, backend=backend_fw)
    ret_from_np = helpers.flatten_and_to_np(ret=ret_from_np, backend=backend_fw)
    for u, v in zip(ret_np, ret_from_np):
        assert u.dtype == v.dtype
        assert u.shape == v.shape


@pytest.mark.xfail
@handle_frontend_test(
    fn_tree="jax.random.poisson",
    dtype_key=helpers.dtype_and_values(
        available_dtypes=["uint32"],
        min_value=0,
        max_value=2000,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
        max_dim_size=2,
    ),
    lam=st.floats(min_value=0, max_value=5, exclude_min=True),
    shape=helpers.get_shape(
        min_num_dims=2, max_num_dims=2, min_dim_size=1, max_dim_size=5
    ),
    dtype=helpers.get_dtypes("integer", full=False),
    test_with_out=st.just(False),
)
def test_jax_poisson(
    *,
    dtype_key,
    lam,
    shape,
    dtype,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, key = dtype_key

    def call():
        return helpers.test_frontend_function(
            input_dtypes=input_dtype,
            backend_to_test=backend_fw,
            frontend=frontend,
            test_flags=test_flags,
            fn_tree=fn_tree,
            on_device=on_device,
            test_values=False,
            key=key[0],
            lam=lam,
            shape=shape,
            dtype=dtype[0],
        )

    ret = call()

    if not ivy.exists(ret):
        return

    ret_np, ret_from_np = ret
    ret_np = helpers.flatten_and_to_np(
        ret=ret_np,
        backend=backend_fw,
    )
    ret_from_np = helpers.flatten_and_to_np(
        ret=ret_from_np,
        backend=backend_fw,
    )
    for u, v in zip(ret_np, ret_from_np):
        assert u.dtype == v.dtype
        assert u.shape == v.shape


@st.composite
def _all_gamma_params(draw):
    shape = draw(
        helpers.get_shape(
            min_dim_size=1, max_dim_size=5, min_num_dims=2, max_num_dims=2
        )
        | st.just(None)
    )
    if shape is None:
        a = draw(
            helpers.array_values(
                min_value=0.0,
                max_value=100.0,
                dtype=helpers.get_dtypes("float", full=False),
                exclude_min=True,
                shape=helpers.get_shape(
                    min_dim_size=1, max_dim_size=5, min_num_dims=1, max_num_dims=2
                ),
            )
        )
        return a[0], shape
    a = draw(st.floats(min_value=0, max_value=5, exclude_min=True))
    return a, shape


@pytest.mark.xfail
@handle_frontend_test(
    fn_tree="jax.random.gamma",
    dtype_key=helpers.dtype_and_values(
        available_dtypes=["uint32"],
        min_value=0,
        max_value=2000,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
        max_dim_size=2,
    ),
    a_shape=_all_gamma_params(),
    dtype=helpers.get_dtypes("float", full=False),
    test_with_out=st.just(False),
)
def test_jax_gamma(
    *,
    dtype_key,
    a_shape,
    dtype,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, key = dtype_key
    a, shape = a_shape

    def call():
        return helpers.test_frontend_function(
            input_dtypes=input_dtype,
            frontend=frontend,
            test_flags=test_flags,
            fn_tree=fn_tree,
            backend_to_test=backend_fw,
            on_device=on_device,
            test_values=False,
            key=key[0],
            a=a,
            shape=shape,
            dtype=dtype[0],
        )

    ret = call()

    if not ivy.exists(ret):
        return

    ret_np, ret_from_np = ret
    ret_np = helpers.flatten_and_to_np(ret=ret_np, backend=backend_fw)
    ret_from_np = helpers.flatten_and_to_np(ret=ret_from_np, backend=backend_fw)
    for u, v in zip(ret_np, ret_from_np):
        assert u.dtype == v.dtype
        assert u.shape == v.shape


@pytest.mark.xfail
@handle_frontend_test(
    fn_tree="jax.random.gumbel",
    dtype_key=helpers.dtype_and_values(
        available_dtypes=["uint32"],
        min_value=0,
        max_value=2000,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
        max_dim_size=2,
    ),
    shape=helpers.get_shape(
        min_dim_size=1, max_num_dims=6, max_dim_size=6, min_num_dims=1, allow_none=False
    ),
    dtype=helpers.get_dtypes("float", full=False),
)
def test_jax_gumbel(
    *,
    dtype_key,
    shape,
    dtype,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, key = dtype_key

    def call():
        return helpers.test_frontend_function(
            input_dtypes=input_dtype,
            frontend=frontend,
            test_flags=test_flags,
            backend_to_test=backend_fw,
            fn_tree=fn_tree,
            on_device=on_device,
            test_values=False,
            key=key[0],
            shape=shape,
            dtype=dtype[0],
        )

    ret = call()

    if not ivy.exists(ret):
        return

    ret_np, ret_from_np = ret
    ret_np = helpers.flatten_and_to_np(ret=ret_np, backend=backend_fw)
    ret_from_np = helpers.flatten_and_to_np(ret=ret_from_np, backend=backend_fw)
    for u, v in zip(ret_np, ret_from_np):
        assert u.dtype == v.dtype
        assert u.shape == v.shape


@pytest.mark.xfail
@handle_frontend_test(
    fn_tree="jax.random.t",
    dtype_key=helpers.dtype_and_values(
        available_dtypes=["uint32"],
        min_value=0,
        max_value=2000,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
        max_dim_size=2,
    ),
    df=st.floats(min_value=0, max_value=5, exclude_min=True),
    shape=helpers.get_shape(
        min_num_dims=1, max_num_dims=6, min_dim_size=1, max_dim_size=6
    ),
    dtype=helpers.get_dtypes("float", full=False),
    test_with_out=st.just(False),
)
def test_jax_t(
    *,
    dtype_key,
    df,
    shape,
    dtype,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, key = dtype_key

    def call():
        return helpers.test_frontend_function(
            input_dtypes=input_dtype,
            frontend=frontend,
            test_flags=test_flags,
            backend_to_test=backend_fw,
            fn_tree=fn_tree,
            on_device=on_device,
            test_values=False,
            key=key[0],
            df=df,
            shape=shape,
            dtype=dtype[0],
        )

    ret = call()

    if not ivy.exists(ret):
        return

    ret_np, ret_from_np = ret
    ret_np = helpers.flatten_and_to_np(ret=ret_np, backend=backend_fw)
    ret_from_np = helpers.flatten_and_to_np(ret=ret_from_np, backend=backend_fw)
    for u, v in zip(ret_np, ret_from_np):
        assert u.dtype == v.dtype
        assert u.shape == v.shape


@pytest.mark.xfail
@handle_frontend_test(
    fn_tree="jax.random.generalized_normal",
    dtype_key=helpers.dtype_and_values(
        available_dtypes=["uint32"],
        min_value=0,
        max_value=2000,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
        max_dim_size=2,
    ),
    p=st.floats(min_value=1e-5, max_value=100, exclude_min=True),
    shape=helpers.get_shape(
        min_num_dims=1, max_num_dims=6, min_dim_size=1, max_dim_size=6
    ),
    dtype=helpers.get_dtypes("float", full=False),
    test_with_out=st.just(False),
)
def test_jax_generalized_normal(
    *,
    dtype_key,
    p,
    shape,
    dtype,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, key = dtype_key

    def call():
        return helpers.test_frontend_function(
            input_dtypes=input_dtype,
            frontend=frontend,
            backend_to_test=backend_fw,
            test_flags=test_flags,
            fn_tree=fn_tree,
            on_device=on_device,
            test_values=False,
            key=key[0],
            p=p,
            shape=shape,
            dtype=dtype[0],
        )

    ret = call()

    if not ivy.exists(ret):
        return

    ret_np, ret_from_np = ret
    ret_np = helpers.flatten_and_to_np(ret=ret_np, backend=backend_fw)
    ret_from_np = helpers.flatten_and_to_np(ret=ret_from_np, backend=backend_fw)
    for u, v in zip(ret_np, ret_from_np):
        assert u.dtype == v.dtype
        assert u.shape == v.shape


@pytest.mark.xfail
@handle_frontend_test(
    fn_tree="jax.random.rademacher",
    dtype_key=helpers.dtype_and_values(
        available_dtypes=["uint32"],
        min_value=0,
        max_value=2000,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
        max_dim_size=2,
    ),
    shape=helpers.get_shape(allow_none=False, min_num_dims=1, min_dim_size=1),
    dtype=helpers.get_dtypes("integer", full=False),
)
def test_jax_rademacher(
    *,
    dtype_key,
    shape,
    dtype,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, key = dtype_key

    def call():
        return helpers.test_frontend_function(
            input_dtypes=input_dtype,
            frontend=frontend,
            backend_to_test=backend_fw,
            test_flags=test_flags,
            fn_tree=fn_tree,
            on_device=on_device,
            test_values=False,
            key=key[0],
            shape=shape,
            dtype=dtype[0],
        )

    ret = call()

    if not ivy.exists(ret):
        return

    ret_np, ret_from_np = ret
    ret_np = helpers.flatten_and_to_np(ret=ret_np, backend=backend_fw)
    ret_from_np = helpers.flatten_and_to_np(ret=ret_from_np, backend=backend_fw)
    for u, v in zip(ret_np, ret_from_np):
        assert u.dtype == v.dtype
        assert u.shape == v.shape


@pytest.mark.xfail
@handle_frontend_test(
    fn_tree="jax.random.randint",
    dtype_key=helpers.dtype_and_values(
        available_dtypes=["uint32"],
        min_value=0,
        max_value=2000,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
        max_dim_size=2,
    ),
    shape=helpers.get_shape(allow_none=False, min_num_dims=1, min_dim_size=1),
    dtype=helpers.get_dtypes("integer", full=False),
    min_max=helpers.general_helpers.get_bounds(dtype="int16"),
)
def test_jax_randint(
    *,
    dtype_key,
    shape,
    dtype,
    min_max,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, key = dtype_key
    minval, maxval = min_max

    def call():
        return helpers.test_frontend_function(
            input_dtypes=input_dtype,
            frontend=frontend,
            test_flags=test_flags,
            backend_to_test=backend_fw,
            fn_tree=fn_tree,
            on_device=on_device,
            test_values=False,
            key=key[0],
            shape=shape,
            minval=minval,
            maxval=maxval,
            dtype=dtype[0],
        )

    ret = call()

    if not ivy.exists(ret):
        return

    ret_np, ret_from_np = ret
    ret_np = helpers.flatten_and_to_np(ret=ret_np, backend=backend_fw)
    ret_from_np = helpers.flatten_and_to_np(ret=ret_from_np, backend=backend_fw)
    for u, v in zip(ret_np, ret_from_np):
        assert u.dtype == v.dtype
        assert u.shape == v.shape


@st.composite
def dtype_p_shape(draw):
    dtype = draw(helpers.array_dtypes(available_dtypes=("float32", "float64")))
    shape = draw(helpers.get_shape(allow_none=False, min_num_dims=1, max_num_dims=3))

    dtype_and_probs = draw(
        helpers.dtype_and_values(
            available_dtypes=dtype, min_value=0, max_value=1, shape=shape
        )
    )
    return dtype_and_probs, shape


@pytest.mark.xfail
@handle_frontend_test(
    fn_tree="jax.random.bernoulli",
    dtype_key=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer", full=False),
        min_value=0,
        max_value=2000,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
        max_dim_size=2,
    ),
    dtype_p_shape_=dtype_p_shape(),
)
def test_jax_bernoulli(
    *,
    dtype_key,
    dtype_p_shape_,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, key = dtype_key
    dtype_p, shape = dtype_p_shape_
    dtype, p = dtype_p

    def call():
        return helpers.test_frontend_function(
            input_dtypes=input_dtype + dtype,
            frontend=frontend,
            test_flags=test_flags,
            fn_tree=fn_tree,
            backend_to_test=backend_fw,
            on_device=on_device,
            test_values=False,
            key=key[0],
            p=p[0],
            shape=shape,
        )

    ret = call()

    if not ivy.exists(ret):
        return

    ret_np, ret_from_np = ret
    ret_np = helpers.flatten_and_to_np(ret=ret_np, backend=backend_fw)
    ret_from_np = helpers.flatten_and_to_np(ret=ret_from_np, backend=backend_fw)
    for u, v in zip(ret_np, ret_from_np):
        assert u.dtype == v.dtype
        assert u.shape == v.shape


@pytest.mark.xfail
@handle_frontend_test(
    fn_tree="jax.random.fold_in",
    dtype_key=helpers.dtype_and_values(
        available_dtypes=["uint32"],
        min_value=0,
        max_value=2000,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
        max_dim_size=2,
    ),
    data=helpers.ints(),
)
def test_jax_fold_in(
    *,
    dtype_key,
    data,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, key = dtype_key

    def call():
        return helpers.test_frontend_function(
            input_dtypes=input_dtype,
            frontend=frontend,
            backend_to_test=backend_fw,
            test_flags=test_flags,
            fn_tree=fn_tree,
            on_device=on_device,
            test_values=False,
            key=key[0],
            data=data,
        )

    ret = call()

    if not ivy.exists(ret):
        return

    ret_np, ret_from_np = ret
    ret_np = helpers.flatten_and_to_np(ret=ret_np, backend=backend_fw)
    ret_from_np = helpers.flatten_and_to_np(ret=ret_from_np, backend=backend_fw)
    for u, v in zip(ret_np, ret_from_np):
        assert u.dtype == v.dtype
        assert u.shape == v.shape


@pytest.mark.xfail
@handle_frontend_test(
    fn_tree="jax.random.permutation",
    dtype_key=helpers.dtype_and_values(
        available_dtypes=["uint32"],
        min_value=0,
        max_value=2000,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
        max_dim_size=2,
    ),
    x=st.integers(min_value=0, max_value=10),
    axis=st.integers(min_value=0, max_value=0),
)
def test_jax_permutation(
    *,
    dtype_key,
    x,
    axis,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, key = dtype_key

    def call():
        return helpers.test_frontend_function(
            input_dtypes=input_dtype,
            frontend=frontend,
            test_flags=test_flags,
            backend_to_test=backend_fw,
            fn_tree=fn_tree,
            on_device=on_device,
            test_values=False,
            key=key[0],
            x=x,
            axis=axis,
        )

    ret = call()

    if not ivy.exists(ret):
        return

    ret_np, ret_from_np = ret
    ret_np = helpers.flatten_and_to_np(ret=ret_np, backend=backend_fw)
    ret_from_np = helpers.flatten_and_to_np(ret=ret_from_np, backend=backend_fw)
    for u, v in zip(ret_np, ret_from_np):
        assert u.dtype == v.dtype
        assert u.shape == v.shape


# loggamma
@pytest.mark.xfail
@handle_frontend_test(
    fn_tree="jax.random.loggamma",
    dtype_key=helpers.dtype_and_values(
        available_dtypes=["uint32"],
        min_value=0,
        max_value=2000,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
        max_dim_size=2,
    ),
    a_shape=_all_gamma_params(),
    dtype=helpers.get_dtypes("float", full=False),
    test_with_out=st.just(False),
)
def test_jax_loggamma(
    *,
    dtype_key,
    a_shape,
    dtype,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, key = dtype_key
    a, shape = a_shape

    def call():
        return helpers.test_frontend_function(
            input_dtypes=input_dtype,
            frontend=frontend,
            backend_to_test=backend_fw,
            test_flags=test_flags,
            fn_tree=fn_tree,
            on_device=on_device,
            test_values=False,
            key=key[0],
            a=a,
            shape=shape,
            dtype=dtype[0],
        )

    ret = call()

    if not ivy.exists(ret):
        return

    ret_np, ret_from_np = ret
    ret_np = helpers.flatten_and_to_np(ret=ret_np, backend=backend_fw)
    ret_from_np = helpers.flatten_and_to_np(ret=ret_from_np, backend=backend_fw)
    for u, v in zip(ret_np, ret_from_np):
        assert u.dtype == v.dtype
        assert u.shape == v.shape


@pytest.mark.xfail
@handle_frontend_test(
    fn_tree="jax.random.shuffle",
    dtype_key=helpers.dtype_and_values(
        available_dtypes=["uint32"],
        min_value=0,
        max_value=2000,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
        max_dim_size=2,
    ),
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=("float32", "float64"),
        valid_axis=True,
        max_axes_size=1,
        force_int_axis=True,
    ),
)
def test_jax_shuffle(
    *,
    dtype_key,
    dtype_x_axis,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    key_dtype, key = dtype_key
    x_dtypes, x, axis = dtype_x_axis

    def call():
        return helpers.test_frontend_function(
            input_dtypes=key_dtype + x_dtypes,
            frontend=frontend,
            backend_to_test=backend_fw,
            test_flags=test_flags,
            fn_tree=fn_tree,
            on_device=on_device,
            test_values=False,
            key=key[0],
            x=x[0],
            axis=axis,
        )

    ret = call()

    if not ivy.exists(ret):
        return

    ret_np, ret_from_np = ret
    ret_np = helpers.flatten_and_to_np(ret=ret_np, backend=backend_fw)
    ret_from_np = helpers.flatten_and_to_np(ret=ret_from_np, backend=backend_fw)
    for u, v in zip(ret_np, ret_from_np):
        assert u.dtype == v.dtype
        assert u.shape == v.shape


@pytest.mark.xfail
@handle_frontend_test(
    fn_tree="jax.random.exponential",
    dtype_key=helpers.dtype_and_values(
        available_dtypes=["uint32"],
        min_value=0,
        max_value=2000,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
        max_dim_size=2,
    ),
    shape=helpers.get_shape(allow_none=False, min_num_dims=1, min_dim_size=1),
    dtype=helpers.get_dtypes("float", full=False),
)
def test_jax_exponential(
    *,
    dtype_key,
    shape,
    dtype,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, key = dtype_key

    def call():
        return helpers.test_frontend_function(
            input_dtypes=input_dtype,
            frontend=frontend,
            backend_to_test=backend_fw,
            test_flags=test_flags,
            fn_tree=fn_tree,
            on_device=on_device,
            test_values=False,
            key=key[0],
            shape=shape,
            dtype=dtype[0],
        )

    ret = call()

    if not ivy.exists(ret):
        return

    ret_np, ret_from_np = ret
    ret_np = helpers.flatten_and_to_np(ret=ret_np, backend=backend_fw)
    ret_from_np = helpers.flatten_and_to_np(ret=ret_from_np, backend=backend_fw)
    for u, v in zip(ret_np, ret_from_np):
        assert u.dtype == v.dtype
        assert u.shape == v.shape


@pytest.mark.xfail
@handle_frontend_test(
    fn_tree="jax.random.weibull_min",
    dtype_key=helpers.dtype_and_values(
        available_dtypes=["uint32"],
        min_value=0,
        max_value=2000,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
        max_dim_size=2,
    ),
    shape=helpers.get_shape(allow_none=False, min_num_dims=1, min_dim_size=1),
    dtype=helpers.get_dtypes("float", full=False),
)
def test_jax_weibull_min(
    *,
    dtype_key,
    shape,
    scale,
    concentration,
    dtype,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, key = dtype_key

    def call():
        return helpers.test_frontend_function(
            input_dtypes=input_dtype,
            frontend=frontend,
            test_flags=test_flags,
            fn_tree=fn_tree,
            backend_to_test=backend_fw,
            on_device=on_device,
            test_values=False,
            key=key[0],
            shape=shape,
            scale=scale,
            concentration=concentration,
            dtype=dtype[0],
        )

    ret = call()

    if not ivy.exists(ret):
        return

    ret_np, ret_from_np = ret
    ret_np = helpers.flatten_and_to_np(ret=ret_np, backend=backend_fw)
    ret_from_np = helpers.flatten_and_to_np(ret=ret_from_np, backend=backend_fw)
    for u, v in zip(ret_np, ret_from_np):
        assert u.dtype == v.dtype
        assert u.shape == v.shape


@st.composite
def get_shape_and_arrays(draw):
    b_shapes = draw(
        helpers.array_and_broadcastable_shape(dtype=helpers.get_dtypes("float"))
    )
    b, shapes = b_shapes
    shapes = draw(st.sampled_from([None, shapes]))
    return b, shapes


@pytest.mark.xfail
@handle_frontend_test(
    fn_tree="jax.random.pareto",
    dtype_key=helpers.dtype_and_values(
        available_dtypes=["uint32"],
        min_value=0,
        max_value=2000,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
        max_dim_size=2,
    ),
    b_shapes=get_shape_and_arrays(),
    dtype=helpers.get_dtypes("float", full=False),
)
def test_jax_pareto(
    *,
    dtype_key,
    b_shapes,
    dtype,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, key = dtype_key
    b, shape = b_shapes

    def call():
        return helpers.test_frontend_function(
            input_dtypes=input_dtype,
            frontend=frontend,
            backend_to_test=backend_fw,
            test_flags=test_flags,
            fn_tree=fn_tree,
            on_device=on_device,
            test_values=False,
            key=key[0],
            b=b,
            shape=shape,
            dtype=dtype[0],
        )

    ret = call()

    if not ivy.exists(ret):
        return
    ret_np, ret_from_np = ret
    if shape is not None:
        assert ret_np.shape == shape
    else:
        assert ret_np.shape == b.shape


@pytest.mark.xfail
@handle_frontend_test(
    fn_tree="jax.random.maxwell",
    dtype_key=helpers.dtype_and_values(
        available_dtypes=["uint32"],
        min_value=0,
        max_value=2000,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
        max_dim_size=2,
    ),
    shape=helpers.get_shape(),
    dtype=helpers.get_dtypes("float", full=False),
)
def test_jax_maxwell(
    *,
    dtype_key,
    shape,
    dtype,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, key = dtype_key

    def call():
        return helpers.test_frontend_function(
            input_dtypes=input_dtype,
            frontend=frontend,
            backend_to_test=backend_fw,
            test_flags=test_flags,
            fn_tree=fn_tree,
            on_device=on_device,
            test_values=False,
            key=key[0],
            shape=shape,
            dtype=dtype[0],
        )

    ret = call()

    if not ivy.exists(ret):
        return

    ret_np, ret_from_np = ret
    ret_np = helpers.flatten_and_to_np(ret=ret_np, backend=backend_fw)
    ret_from_np = helpers.flatten_and_to_np(ret=ret_from_np, backend=backend_fw)
    for u, v in zip(ret_np, ret_from_np):
        assert u.dtype == v.dtype
        assert u.shape == v.shape


@pytest.mark.xfail
@handle_frontend_test(
    fn_tree="jax.random.ball",
    dtype_key=helpers.dtype_and_values(
        available_dtypes=["uint32"],
        min_value=0,
        max_value=2000,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
        max_dim_size=2,
    ),
    shape=helpers.get_shape(
        min_num_dims=1, max_num_dims=6, min_dim_size=1, max_dim_size=6
    ),
    dtype=helpers.get_dtypes("float", full=False),
    d=st.integers(min_value=1, max_value=100),
    p=st.floats(min_value=1e-5, max_value=100, exclude_min=True),
    test_with_out=st.just(False),
)
def test_jax_ball(
    *,
    dtype_key,
    d,
    p,
    shape,
    dtype,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, key = dtype_key

    def call():
        return helpers.test_frontend_function(
            input_dtypes=input_dtype,
            frontend=frontend,
            test_flags=test_flags,
            backend_to_test=backend_fw,
            fn_tree=fn_tree,
            on_device=on_device,
            test_values=False,
            key=key[0],
            d=d,
            p=p,
            shape=shape,
            dtype=dtype[0],
        )

    ret = call()

    if not ivy.exists(ret):
        return

    ret_np, ret_from_np = ret
    ret_np = helpers.flatten_and_to_np(ret=ret_np, backend=backend_fw)
    ret_from_np = helpers.flatten_and_to_np(ret=ret_from_np, backend=backend_fw)
    for u, v in zip(ret_np, ret_from_np):
        assert u.dtype == v.dtype
        assert u.shape == v.shape
