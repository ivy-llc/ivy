# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


@handle_frontend_test(
    fn_tree="numpy.roots",
    dtype_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=1,
        min_value=2,
    ),
    test_with_out=st.just(False),
)
def test_numpy_roots(dtype_values, frontend, backend_fw, test_flags, fn_tree):
    input_dtypes, values = dtype_values
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        fn_tree=fn_tree,
        backend_to_test="torch",
        frontend=frontend,
        test_flags=test_flags,
        p=values[0],
    )
