# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


@handle_frontend_test(
    fn_tree="paddle.tensor.random.exponential_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=1000,
        min_num_dims=1,
        max_num_dims=10,
        min_dim_size=2,
        max_dim_size=10,
    ),
)
def test_paddle_exponential_(
    fn_tree,
    dtype_and_x,
    frontend,
    backend_fw,
    test_flags,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        test_values=False,
        x=x[0],
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
    backend_fw,
    test_flags,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        test_values=False,
        x=x[0],
        min=min,
        max=max,
        seed=seed,
    )
