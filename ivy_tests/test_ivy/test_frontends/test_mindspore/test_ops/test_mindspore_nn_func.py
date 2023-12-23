# global
import pytest

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# softsign
@pytest.mark.skip("Testing pipeline not yet implemented")
@handle_frontend_test(
    fn_tree="mindspore.ops.softsign",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes(kind="float", full=False, key="dtype"),
        safety_factor_scale="log",
        small_abs_safety_factor=20,
    ),
)
def test_mindspore_softsign(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )
