
# global
from hypothesis import strategies as st


# local
from ivy_tests.test_ivy.test_functional.test_core.test_statistical import (
    _statistical_dtype_values,
)
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy import numpy as ivy_np
from ivy_tests.helpers import assert_allclose
from ivy_tests.frontend_helpers import (
    handle_frontend_test,
    handle_frontend_method,
    get_dtypes,
    dtype_and_values,
    get_shape,
    where,
    test_frontend_function,
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

@handle_frontend_test(
    fn_tree="ivy_np.quantile",
    dtypes_values_casting=dtype_and_values(
        available_dtypes=get_dtypes("float"),
    ),
    where=where(),
    number_positional_args=2,
)
def test_ivy_quantile(
    dtypes_values_casting,
    where,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtypes, x = dtypes_values_casting

    # Prepare the inputs for testing
    arr = x[0]
    q = 50.0  # Quantile value

    # Calculate the expected result using NumPy
    expected_result = ivy_np.quantile(arr, q, axis=None, interpolation='linear')

    # Apply the quantile function using the Ivy frontend
    result = frontend.quantile(arr, q, axis=None, interpolation='linear')

    # Assert that the Ivy result matches the expected result
    assert_allclose(result, expected_result)
