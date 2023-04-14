from turtle import st
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.test_functional.test_core.test_statistical import (
    statistical_dtype_values,
)
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test

from ivy_tests.test_ivy.helpers import handle_frontend_test


#bincount

@handle_frontend_test(
    fn_tree="numpy.bincount",

    dtype_x_axis = statistical_dtype_values(function="bincount"),
    dtype = helpers.get_dtypes("integer", full=False, none=True),
    where = np_frontend_helpers.where(),
    keep_dims = st.booleans(),
)
def test_numpy_bincount(
    *,
    dtype_and_x,
    minlength,
    weights,
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
        x=x[0],
        weights=weights,
        minlength=minlength,

    )