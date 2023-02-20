# global
from hypothesis import strategies as st, assume


# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.test_functional.test_core.test_statistical import (
    statistical_dtype_values,
)
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test

# percentile
@handle_frontend_test(
    fn_tree="numpy.percentile",
    dtype_and_x=np_frontend_helpers.dtypes_values_casting_dtype(
        arr_func=[
            lambda: helpers.dtype_and_values(
                available_dtypes=helpers.get_dtypes("numeric"),
                num_arrays=2,
                large_abs_safety_factor=4,
                shared_dtype=True,
                safety_factor_scale="linear",
            )
        ],
        get_dtypes_kind="numeric",
    ),
    dtype=helpers.get_dtypes("float", full=False, none=True),
    where=np_frontend_helpers.where(),
    keep_dims=st.booleans(),
)
def test_numpy_percentile(
    dtype_and_x,
    dtype,
    where,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    keep_dims,
):
    input_dtypes, x, axis = dtype_and_x
    if isinstance(axis, tuple):
        axis = axis[0]

    where, input_dtypes, test_flags = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )

    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        q=x[1],
        axis=axis,
        out=None,
        keepdims=keep_dims,
        where=where,
        test_values=False,
    )
