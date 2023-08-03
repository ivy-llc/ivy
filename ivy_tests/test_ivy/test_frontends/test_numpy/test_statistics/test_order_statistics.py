# global
from hypothesis import strategies as st


# local
from ivy_tests.test_ivy.test_functional.test_core.test_statistical import (
    _statistical_dtype_values,
)
import ivy_tests.test_ivy.helpers as helpers
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# ptp
@handle_frontend_test(
    fn_tree="numpy.ptp",
    dtype_values_axis=_statistical_dtype_values(function="ptp"),
    keep_dims=st.booleans(),
    test_with_out=st.just(False),
)
def test_numpy_ptp(
    dtype_values_axis,
    frontend,
    test_flags,
    fn_tree,
    keep_dims,
):
    input_dtypes, values, axis = dtype_values_axis
    if isinstance(axis, tuple):
        axis = axis[0]

    helpers.test_frontend_function(
        a=values[0],
        axis=axis,
        out=None,
        keepdims=keep_dims,
        frontend=frontend,
        fn_tree=fn_tree,
        test_flags=test_flags,
        input_dtypes=input_dtypes,
    )


# nanpercentile
@handle_frontend_test(
    fn_tree="numpy.nanpercentile",
    dtype_values_axis=_statistical_dtype_values(function="nanpercentile"),
    where=np_frontend_helpers.where(),
    keep_dims=st.booleans(),
)
def test_numpy_nanpercentile(
    dtype_values_axis,
    where,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    keep_dims,
):
    input_dtypes, values, axis = dtype_values_axis
    if isinstance(axis, tuple):
        axis = axis[0]

    where, input_dtypes, test_flags = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )

    np_frontend_helpers.test_frontend_function(
        a=values[0][0],
        q=values[0][1],
        axis=axis,
        out=None,
        overwrite_input=None,
        method=None,
        keepdims=keep_dims,
        interpolation=None,
        frontend=frontend,
        fn_tree=fn_tree,
        test_flags=test_flags,
        input_dtypes=input_dtypes,
    )
