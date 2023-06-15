# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


@handle_frontend_test(
    fn_tree="paddle.uniform",
    input_dtypes=helpers.get_dtypes("float"),
    shape=st.tuples(
        st.integers(min_value=2, max_value=5), st.integers(min_value=2, max_value=5)
    ),
    dtype=helpers.get_dtypes("valid", full=False),
    min=st.floats(allow_nan=False, allow_infinity=False, width=32),
    max=st.floats(allow_nan=False, allow_infinity=False, width=32),
    seed=st.integers(min_value=2, max_value=5),
)
def test_paddle_uniform(
    input_dtypes,
    shape,
    dtype,
    min,
    max,
    seed,
    frontend,
    test_flags,
    fn_tree,
):
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        test_values=False,
        shape=shape,
        dtype=dtype[0],
        min=min,
        max=max,
        seed=seed,
    )


@handle_frontend_test(
    fn_tree="paddle.randn",
    input_dtypes=helpers.get_dtypes("valid"),
    shape=st.tuples(
        st.integers(min_value=2, max_value=5), st.integers(min_value=2, max_value=5)
    ),
    dtype=helpers.get_dtypes("valid", full=False),
    seed=st.integers(min_value=0, max_value=5),
)
def test_paddle_randn(
    input_dtypes,
    shape,
    dtype,
    seed,
    frontend,
    test_flags,
    fn_tree,
):
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        test_values=False,
        shape=shape,
        dtype=dtype[0],
        seed=seed,
    )
