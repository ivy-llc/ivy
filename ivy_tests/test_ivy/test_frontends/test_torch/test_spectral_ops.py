import ivy_tests.test_ivy.helpers as helpers
from hypothesis import strategies as st
from ivy_tests.test_ivy.helpers.testing_helpers import handle_frontend_test


@handle_frontend_test(
    fn_tree="torch.hann_window",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("int")),
    periodic=st.booleans(),
    dtype=helpers.get_dtypes("float", full=False, none=True),
)
def test_torch_hann_window(
    periodic, dtype_and_x, dtype, fn_tree, frontend, on_device, test_flags
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        on_device=on_device,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        periodic=periodic,
        dtype=dtype[0],
        window_length=x[0],
    )
