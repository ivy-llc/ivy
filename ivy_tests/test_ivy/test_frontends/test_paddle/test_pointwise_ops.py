# global
import numpy as np
from hypothesis import strategies as st, assume

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# sin
@handle_frontend_test(
    fn_tree="paddle.sin",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_paddle_sin(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
    )
