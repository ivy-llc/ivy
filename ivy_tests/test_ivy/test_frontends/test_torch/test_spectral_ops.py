# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# hamming_window
@handle_frontend_test(
    fn_tree="torch.hamming_window",
    window_length=helpers.ints(min_value=1),
    periodic=st.booleans(),
    alpha=helpers.floats(min_value=0),
    beta=helpers.floats(min_value=0),
    requires_grad=st.booleans(),
    test_with_out=st.just(False),
    dtype=helpers.get_dtypes("integer"),
)
def test_torch_hamming_window(
    window_length,
    periodic,
    alpha,
    beta,
    *,
    dtype,
    requires_grad,
    fn_tree,
    frontend,
    test_flags,
):
    helpers.test_frontend_function(
        input_dtypes=dtype,
        window_length=window_length,
        periodic=periodic,
        alpha=alpha,
        beta=beta,
        requires_grad=requires_grad,
        fn_tree=fn_tree,
        frontend=frontend,
        test_flags=test_flags,
    )
