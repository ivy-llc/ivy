# global
from hypothesis import strategies as st

# local

import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# randint
@handle_frontend_test(
    fn_tree="paddle.randint",
    low=helpers.ints(min_value=0, max_value=10),
    high=helpers.ints(min_value=11, max_value=20),
    dtype=helpers.get_dtypes("integer"),
    shape=helpers.get_shape(
        allow_none=False, min_num_dims=2, max_num_dims=7, min_dim_size=2
    ),
)
def test_paddle_randint(
    low,
    high,
    dtype,
    frontend,
    test_flags,
    shape,
    fn_tree,
):
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_values=False,
        fn_tree=fn_tree,
        test_flags=test_flags,
        low=low,
        high=high,
        shape=shape,
    )


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
    fn_tree="paddle.poisson",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=1000,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
        max_dim_size=2,
    ),
)
def test_paddle_poisson(dtype_and_x, frontend, test_flags, fn_tree):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        test_values=False,
        x=x[0],
    )


@handle_frontend_test(
    fn_tree="paddle.rand",
    input_dtypes=st.sampled_from(["int32", "int64"]),
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=0,
        min_dim_size=1,
    ),
    dtype=helpers.get_dtypes("valid", full=False),
)
def test_paddle_rand(
    *,
    input_dtypes,
    shape,
    dtype,
    frontend,
    test_flags,
    fn_tree,
):
    helpers.test_frontend_function(
        input_dtypes=[input_dtypes],
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        test_values=False,
        shape=shape,
        dtype=dtype[0],
    )


@handle_frontend_test(
    fn_tree="paddle.randn",
    input_dtypes=st.sampled_from(["int32", "int64"]),
    shape=helpers.get_shape(
        allow_none=False, min_num_dims=1, max_num_dims=1, min_dim_size=2
    ),
    dtype=st.sampled_from(["float32", "float64"]),
)
def test_paddle_randn(
    *,
    input_dtypes,
    shape,
    dtype,
    frontend,
    test_flags,
    fn_tree,
):
    helpers.test_frontend_function(
        input_dtypes=[input_dtypes],
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        test_values=False,
        shape=shape,
        dtype=dtype,
    )


@handle_frontend_test(
    fn_tree="paddle.tensor.random.uniform_",
    min=helpers.floats(min_value=-1, max_value=0),
    max=helpers.floats(min_value=0.1, max_value=1),
    seed=st.integers(min_value=2, max_value=5),
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=1000,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
        max_dim_size=2,
    ),
)
def test_paddle_uniform_(
    fn_tree,
    min,
    max,
    seed,
    dtype_and_x,
    frontend,
    test_flags,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        test_values=False,
        x=x[0],
        min=min,
        max=max,
        seed=seed,
    )


@handle_frontend_test(
    fn_tree="paddle.standard_normal",
    input_dtypes=st.sampled_from([["int32"], ["int64"]]),
    shape=helpers.get_shape(
        min_num_dims=1,
        min_dim_size=1,
    ),
    dtype=helpers.get_dtypes("valid", full=False),
)
def test_paddle_standard_normal(
    input_dtypes,
    shape,
    dtype,
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
    )


@handle_frontend_test(
    fn_tree="paddle.randint_like",
    input_dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=helpers.get_shape(
        allow_none=False, min_num_dims=2, max_num_dims=7, min_dim_size=2
        ),
    ),

    low=st.integers(min_value=0, max_value=10),
    high=st.integers(min_value=11, max_value=20),
    dtype=helpers.get_dtypes("integer"),
)
def test_paddle_randint_like(
    input_dtype_and_x,
    low,
    high,
    dtype,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = input_dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        low=low,
        high=high,
        dtype=dtype[0]
    )
