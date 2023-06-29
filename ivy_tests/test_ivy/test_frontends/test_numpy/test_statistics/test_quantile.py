# global
from hypothesis import strategies as st


# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_tests.test_ivy.test_functional.test_core.test_statistical import (
    _statistical_dtype_values,
)


@handle_frontend_test(
    fn_tree="numpy.quantile",
    dtype_and_x=_statistical_dtype_values(function="quantile"),
    # dtype=helpers.get_dtypes("float", full=False, none=True),
    # where=np_frontend_helpers.where(),
    keep_dims=st.booleans(),
)
def test_quantile(dtype_and_x, frontend, test_flags, fn_tree, on_device, keep_dims):
    input_dtype, x, axis, interpolation, q = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        axis=axis,
        out=None,
        keepdims=keep_dims,
        test_values=False,
        a=x[0],
        q=q,
        interpolation=interpolation[0],
    )
