import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.test_frontends.test_torch.test_tensor import _requires_grad
from hypothesis import strategies as st
from ivy_tests.test_ivy.helpers.testing_helpers import handle_frontend_test


@handle_frontend_test(
    fn_tree="torch.hann_window",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric")
    ),
    periodic=st.booleans(),
    dtype=helpers.get_dtypes("numeric", full=False, none=True),
    requires_grad=_requires_grad(),
)
def test_torch_hann_window(
    periodic,
    dtype_and_x,
    dtype,
    num_positional_args: 2,
    on_device,
    frontend,
    requires_grad,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        frontend=frontend,
        window_length=x[0],
        periodic=periodic,
        dtype=dtype[0],
        on_device=on_device,
        requires_grad=requires_grad,
    )
