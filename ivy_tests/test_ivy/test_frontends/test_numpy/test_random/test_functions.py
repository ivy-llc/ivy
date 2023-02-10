# global,
from hypothesis import strategies as st
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
