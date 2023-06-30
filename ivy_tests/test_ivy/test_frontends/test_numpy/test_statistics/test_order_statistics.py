
# global
from hypothesis import strategies as st


# local
from ivy_tests.test_ivy.test_functional.test_core.test_statistical import (
    _statistical_dtype_values,
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
    dtype_values_axis=_statistical_dtype_values(function="quantile"),
    where=np_frontend_helpers.where(),
    keep_dims=st.booleans(),
    
)
def test_numpy_quantile(
    dtype_values_axis,
    where,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    keep_dims,
):
   
    (input_dtypes, values, axis) = dtype_values_axis
    if isinstance(axis, tuple):
        axis = axis[0]

    (
        where,
        input_dtypes,
        test_flags,
    ) = np_frontend_helpers.handle_where_and_array_bools(
        where=where, input_dtype=input_dtypes, test_flags=test_flags
    )

    try:
        a = values[0][0]
        q = values[0][1]

        q = np.clip(q, 0.0, 1.0)

        np_frontend_helpers.test_frontend_function(
            a=a,
            q=q,
            axis=axis,
            out=None,
            overwrite_input=None,
            interpolation="linear",
            keepdims=keep_dims,
            frontend=frontend,
            fn_tree=fn_tree,
            test_flags=test_flags,
            input_dtypes=input_dtypes
            )
    except Exception as e:
        print(f"Error occurred: {e}")
        # Handle the error here, such as logging, reporting, or specific error handling
