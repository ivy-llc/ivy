# global,
from hypothesis import strategies as st, assume
import numpy as np

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# random_sample
@handle_frontend_test(
    fn_tree="numpy.random.random_sample",
    input_dtypes=helpers.get_dtypes("integer", full=False),
    size=helpers.get_shape(allow_none=True),
    test_with_out=st.just(False),
)
def test_numpy_random_sample(
    input_dtypes,
    size,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        size=size,
    )


# dirichlet
@handle_frontend_test(
    fn_tree="numpy.random.dirichlet",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=st.tuples(
            st.integers(min_value=2, max_value=5),
        ),
        min_value=1,
        max_value=100,
        exclude_min=True,
    ),
    size=st.tuples(
        st.integers(min_value=2, max_value=5), st.integers(min_value=2, max_value=5)
    ),
    test_with_out=st.just(False),
)
def test_numpy_dirichlet(
    dtype_and_x,
    size,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        alpha=x[0],
        test_values=False,
        size=size,
    )


# uniform
@handle_frontend_test(
    fn_tree="numpy.random.uniform",
    input_dtypes=helpers.get_dtypes("float", index=2),
    low=st.floats(allow_nan=False, allow_infinity=False, width=32),
    high=st.floats(allow_nan=False, allow_infinity=False, width=32),
    size=st.tuples(
        st.integers(min_value=2, max_value=5), st.integers(min_value=2, max_value=5)
    ),
)
def test_numpy_uniform(
    input_dtypes,
    size,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    low,
    high,
):
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        low=low,
        high=high,
        size=size,
    )


# normal
@handle_frontend_test(
    fn_tree="numpy.random.normal",
    input_dtypes=helpers.get_dtypes("float", index=2),
    loc=st.floats(allow_nan=False, allow_infinity=False, width=32),
    scale=st.floats(allow_nan=False, allow_infinity=False, width=32, min_value=0),
    size=st.tuples(
        st.integers(min_value=2, max_value=5), st.integers(min_value=2, max_value=5)
    ),
)
def test_numpy_normal(
    input_dtypes,
    size,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    loc,
    scale,
):
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        loc=loc,
        scale=scale,
        size=size,
    )


# poisson
@handle_frontend_test(
    fn_tree="numpy.random.poisson",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=st.tuples(st.integers(min_value=1, max_value=2)),
        min_value=1,
        max_value=100,
    ),
    size=st.tuples(
        st.integers(min_value=1, max_value=10), st.integers(min_value=2, max_value=2)
    ),
)
def test_numpy_poisson(
    dtype_and_x,
    size,
    test_flags,
    frontend,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        lam=x[0],
        test_values=False,
        size=size,
    )


@handle_frontend_test(
    fn_tree="numpy.random.geometric",
    input_dtypes=helpers.get_dtypes("float"),
    p=st.floats(
        allow_nan=False,
        allow_infinity=False,
        width=32,
        min_value=9.999999747378752e-06,
        max_value=0.9999899864196777,
    ),
    size=st.tuples(
        st.integers(min_value=2, max_value=5), st.integers(min_value=2, max_value=5)
    ),
)
def test_numpy_geometric(
    input_dtypes,
    size,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    p,
):
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        p=p,
        size=size,
    )


# multinomial
@handle_frontend_test(
    fn_tree="numpy.random.multinomial",
    n=helpers.ints(min_value=2, max_value=10),
    dtype=helpers.get_dtypes("float", full=False),
    size=st.tuples(
        st.integers(min_value=1, max_value=10), st.integers(min_value=2, max_value=2)
    ),
)
def test_numpy_multinomial(
    n,
    dtype,
    size,
    test_flags,
    frontend,
    fn_tree,
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        n=n,
        pvals=np.array([1 / n] * n, dtype=dtype[0]),
        size=size,
    )


# permutation
@handle_frontend_test(
    fn_tree="numpy.random.permutation",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"), min_num_dims=1
    ),
)
def test_numpy_permutation(
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        x=x[0],
    )


# beta
@handle_frontend_test(
    fn_tree="numpy.random.beta",
    input_dtypes=helpers.get_dtypes("float", index=2),
    a=st.floats(
        allow_nan=False, allow_infinity=False, width=32, min_value=0, exclude_min=True
    ),
    b=st.floats(
        allow_nan=False, allow_infinity=False, width=32, min_value=0, exclude_min=True
    ),
    size=st.tuples(
        st.integers(min_value=2, max_value=5), st.integers(min_value=2, max_value=5)
    ),
)
def test_numpy_beta(
    input_dtypes,
    size,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    a,
    b,
):
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        a=a,
        b=b,
        size=size,
    )


# shuffle
@handle_frontend_test(
    fn_tree="numpy.random.shuffle",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"), min_num_dims=1
    ),
)
def test_numpy_shuffle(
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        x=x[0],
    )


@handle_frontend_test(
    fn_tree="numpy.random.standard_normal",
    input_dtypes=helpers.get_dtypes("integer", full=False),
    size=helpers.get_shape(allow_none=True),
    test_with_out=st.just(False),
)
def test_numpy_standard_normal(
    input_dtypes,
    size,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        size=size,
    )


@handle_frontend_test(
    fn_tree="numpy.random.standard_gamma",
    shape_dtypes=helpers.get_dtypes("float", full=False),
    shape=st.floats(
        allow_nan=False, allow_infinity=False, width=32, min_value=0, exclude_min=True
    ),
    size=st.tuples(
        st.integers(min_value=2, max_value=5), st.integers(min_value=2, max_value=5)
    ),
    size_dtypes=helpers.get_dtypes("integer", full=False),
    test_with_out=st.just(False),
)
def test_numpy_standard_gamma(
    shape,
    shape_dtypes,
    size,
    size_dtypes,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    assume("float16" not in shape_dtypes)
    helpers.test_frontend_function(
        input_dtypes=shape_dtypes + size_dtypes,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        shape=shape,
        size=size,
    )


# binomial
@handle_frontend_test(
    fn_tree="numpy.random.binomial",
    n=st.integers(min_value=0, max_value=2),
    dtype=helpers.get_dtypes("float", full=False, index=2),
    p=st.floats(
        allow_nan=False, allow_infinity=False, width=32, min_value=0, max_value=1
    ),
    size=st.tuples(
        st.integers(min_value=2, max_value=5), st.integers(min_value=2, max_value=5)
    ),
)
def test_numpy_binomial(
    dtype,
    size,
    test_flags,
    frontend,
    fn_tree,
    on_device,
    n,
    p,
):
    helpers.test_frontend_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        n=n,
        p=p,
        size=size,
    )


# chisquare
# The test values are restricted to (0, 1000] because df<=0 is invalid
# and very large df can cause problems with type conversions
@handle_frontend_test(
    fn_tree="numpy.random.chisquare",
    dtypes=helpers.get_dtypes("float", full=False),
    df=st.one_of(
        st.floats(
            min_value=0,
            max_value=1000,
            exclude_min=True,
            allow_subnormal=False,
            width=32,
        ),
        st.integers(min_value=1, max_value=1000),
        st.lists(
            st.one_of(
                st.floats(
                    min_value=0,
                    max_value=1000,
                    exclude_min=True,
                    allow_subnormal=False,
                    width=32,
                )
                | st.integers(min_value=1, max_value=1000)
            ),
            min_size=1,
        ),
    ),
    size=helpers.get_shape(allow_none=True),
    test_with_out=st.just(False),
)
def test_numpy_chisquare(
    dtypes,
    df,
    size,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    # make sure `size` is something `df` can be broadcast to
    if (
        hasattr(df, "__len__")
        and size is not None
        and (len(size) == 0 or size[-1] != len(df))
    ):
        size = (*size, len(df))
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        df=df,
        size=size,
    )


# lognormal
# min value is set 0
@handle_frontend_test(
    fn_tree="numpy.random.lognormal",
    input_dtypes=helpers.get_dtypes("float", index=2),
    mean=st.floats(
        allow_nan=False, allow_infinity=False, width=32, min_value=-5, max_value=5
    ),
    sigma=st.floats(
        allow_nan=False, allow_infinity=False, width=32, min_value=0, max_value=5
    ),
    size=st.tuples(
        st.integers(min_value=2, max_value=5), st.integers(min_value=2, max_value=5)
    ),
)
def test_numpy_lognormal(
    input_dtypes,
    size,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    mean,
    sigma,
):
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        mean=mean,
        sigma=sigma,
        size=size,
    )


# negative_binomial
@handle_frontend_test(
    fn_tree="numpy.random.negative_binomial",
    input_dtypes=helpers.get_dtypes("float", index=2),
    # max value for n and min value for p are restricted in testing
    # as they can blow up poisson lambda, which will cause an
    # error (lam value too large).
    n=st.floats(
        allow_nan=False,
        allow_infinity=False,
        width=32,
        min_value=0,
        max_value=100000,
        exclude_min=True,
    ),
    p=st.floats(
        allow_nan=False,
        allow_infinity=False,
        width=32,
        min_value=9.999999747378752e-06,
        exclude_min=True,
        max_value=1,
        exclude_max=True,
    ),
    size=helpers.get_shape(allow_none=True),
    test_with_out=st.just(False),
)
def test_numpy_negative_binomial(
    input_dtypes,
    size,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    n,
    p,
):
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        n=n,
        p=p,
        size=size,
    )


# weibull
@handle_frontend_test(
    fn_tree="numpy.random.weibull",
    input_dtypes=helpers.get_dtypes("float", index=2),
    a=st.floats(
        allow_nan=False,
        allow_infinity=False,
        width=32,
        min_value=1,
        max_value=1000,
        exclude_min=True,
    ),
    size=helpers.get_shape(allow_none=True),
    test_with_out=st.just(False),
)
def test_numpy_weibull(
    input_dtypes,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    a,
    size,
):
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        a=a,
        size=size,
    )


# standard_cauchy
@handle_frontend_test(
    fn_tree="numpy.random.standard_cauchy",
    input_dtypes=helpers.get_dtypes("integer", full=False),
    size=helpers.get_shape(allow_none=True),
    test_with_out=st.just(False),
)
def test_numpy_standard_cauchy(
    input_dtypes,
    size,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        size=size,
    )


# rayleigh
@handle_frontend_test(
    fn_tree="numpy.random.rayleigh",
    input_dtypes=helpers.get_dtypes("float"),
    scale=st.floats(
        allow_nan=False,
        allow_infinity=False,
        width=32,
        min_value=0,
        max_value=1000,
        exclude_min=True,
    ),
    size=helpers.get_shape(allow_none=True),
)
def test_numpy_rayleigh(
    input_dtypes,
    size,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    scale,
):
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        scale=scale,
        size=size,
    )


# gumbel
@handle_frontend_test(
    fn_tree="numpy.random.gumbel",
    input_dtypes=helpers.get_dtypes("float"),
    loc=st.floats(
        allow_nan=False,
        allow_infinity=False,
        width=32,
        max_value=1000,
    ),
    scale=st.floats(
        allow_nan=False,
        allow_infinity=False,
        width=32,
        min_value=0,
        max_value=1000,
        exclude_min=True,
    ),
    size=helpers.get_shape(allow_none=True),
)
def test_numpy_gumbel(
    input_dtypes,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    loc,
    scale,
    size,
):
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        loc=loc,
        scale=scale,
        size=size,
    )


@handle_frontend_test(
    fn_tree="numpy.random.gamma",
    input_dtypes=helpers.get_dtypes("float", full=False),
    shape=st.floats(
        allow_nan=False, allow_infinity=False, width=32, min_value=0, exclude_min=True
    ),
    scale=st.floats(
        allow_nan=False, allow_infinity=False, width=32, min_value=0, exclude_min=True
    ),
    size=st.tuples(
        st.integers(min_value=2, max_value=5), st.integers(min_value=2, max_value=5)
    ),
    test_with_out=st.just(False),
)
def test_numpy_gamma(
    input_dtypes,
    shape,
    scale,
    size,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        shape=shape,
        scale=scale,
        size=size,
    )


@handle_frontend_test(
    fn_tree="numpy.random.triangular",
    input_dtypes=helpers.get_dtypes("float"),
    left=st.floats(
        allow_nan=False,
        allow_infinity=False,
        width=32,
        min_value=0,
        max_value=10,
    ),
    mode=st.floats(
        allow_nan=False,
        allow_infinity=False,
        width=32,
        min_value=10,
        max_value=100,
        exclude_min=True,
    ),
    right=st.floats(
        allow_nan=False,
        allow_infinity=False,
        width=32,
        min_value=100,
        max_value=1000,
        exclude_min=True,
    ),
    size=helpers.get_shape(allow_none=False),
)
def test_numpy_triangular(
    input_dtypes,
    size,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    left,
    mode,
    right,
):
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        size=size,
        left=left,
        right=right,
        mode=mode,
    )
