from ivy.core.container import Container
from ivy.core.operations import arithmetic_ops
from ivy.core.tensor import Tensor
from ivy.numpy.base import promote_types_of_numpy_inputs
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

def quantile(arr, q, axis=None, out=None, overwrite_input=False, interpolation='linear'):
    arr = promote_types_of_numpy_inputs(arr)
    if axis is None:
        arr = arr.flatten()
        axis = 0
    elif axis < 0:
        axis += arr.ndim
    if out is not None:
        out = promote_types_of_numpy_inputs(out)

    # Sort the array along the specified axis
    sorted_arr = arithmetic_ops.sort(arr, axis=axis)

    # Calculate the index based on the desired quantile
    alpha = q / 100.0
    if interpolation == 'linear':
        index = alpha * (sorted_arr.shape[axis] - 1)
        lower_idx = arithmetic_ops.floor(index)
        upper_idx = arithmetic_ops.ceil(index)
        weight = index - lower_idx
    elif interpolation == 'lower':
        index = alpha * sorted_arr.shape[axis]
        lower_idx = arithmetic_ops.floor(index)
        upper_idx = lower_idx
        weight = 0
    elif interpolation == 'higher':
        index = alpha * sorted_arr.shape[axis]
        lower_idx = arithmetic_ops.ceil(index)
        upper_idx = lower_idx
        weight = 0
    else:
        raise ValueError("Interpolation method '{}' not supported.".format(interpolation))

    # Calculate the quantile value
    if out is not None:
        ret = out
    else:
        ret = Container(arr)
    ret.val = (1.0 - weight) * sorted_arr.take(lower_idx, axis=axis) + weight * sorted_arr.take(upper_idx, axis=axis)

    if overwrite_input:
        arr.val = ret.val
        return arr
    else:
        return ret

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

