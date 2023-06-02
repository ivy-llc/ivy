# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_tests.test_ivy.test_functional.test_core.test_statistical import (
    statistical_dtype_values,
)

# std
@handle_frontend_test(
    fn_tree="paddle.std",
    dtype_and_x=statistical_dtype_values(function="std"),
    keepdims=st.booleans(),
)
def test_paddle_std(
    *,
    dtype_and_x,
    keepdims,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x, axis, correction = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        dim=axis,
        unbiased=bool(correction),
        keepdim=keepdims,
    )
