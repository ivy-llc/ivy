import ivy_tests.test_ivy.helpers as helpers
from hypothesis import strategies as st
from ivy_tests.test_ivy.helpers.testing_helpers import handle_frontend_test
from ivy_tests.test_ivy.test_frontends.test_torch.test_tensor import _requires_grad


@handle_frontend_test(
    fn_tree="torch.hann_window",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("int")),
    periodic=st.booleans(),
    dtype=helpers.get_dtypes("float", full=False, none=True),
    requires_grad=_requires_grad(),
)
def test_torch_hann_window(
    periodic,
    dtype_and_x,
    dtype,
    fn_tree,
    frontend,
    test_flags,
    requires_grad,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        requires_grad=requires_grad,
        periodic=periodic,
        dtype=dtype[0],
        window_length=x[0],
    )
