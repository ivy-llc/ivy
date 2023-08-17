# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test


@handle_test(
    fn_tree="functional.ivy.experimental.optional_get_element",
    dtype_and_x=helpers.dtype_and_values(),
)
def test_optional_get_element(
    *,
    dtype_and_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    input_dtype, x = dtype_and_x

    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x=x[0],
    )
