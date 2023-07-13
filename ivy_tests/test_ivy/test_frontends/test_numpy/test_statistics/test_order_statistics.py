# global
from hypothesis import strategies as st


# local
from ivy_tests.test_ivy.test_functional.test_core.test_statistical import (
    _statistical_dtype_values,
)
from ivy_tests.test_ivy.test_functional.test_experimental.test_core.test_statistical import (  # noqa
    _quantile_helper,
)
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


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


@handle_frontend_test(
    fn_tree="numpy.quantile",
    dtype_and_x=_quantile_helper(),
    keepdims=st.booleans(),
    test_with_out=st.just(False),
)
def test_numpy_quantile(
    *,
    dtype_and_x,
    keepdims,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x, axis, method, q = dtype_and_x

    if type(axis) is tuple:
        axis = axis[0]

    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        q=q,
        axis=axis,
        method=method[0],
        keepdims=keepdims,
    )