# global
from hypothesis import given

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy.functional.frontends.torch.func_wrapper import (
    inputs_to_ivy_arrays,
    outputs_to_frontend_arrays,
    to_ivy_arrays_and_back,
)
from ivy.functional.frontends.torch.tensor import Tensor


def _fn(x):
    return x


@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
)
def test_inputs_to_ivy_arrays(dtype_and_x):
    x_dtype, x = dtype_and_x

    # check for ivy array
    input_ivy = ivy.array(x[0], dtype=x_dtype[0])
    output = inputs_to_ivy_arrays(_fn)(input_ivy)
    assert isinstance(output, ivy.Array)
    assert input_ivy.dtype == output.dtype
    assert ivy.all(input_ivy == output)

    # check for native array
    input_native = ivy.native_array(input_ivy)
    output = inputs_to_ivy_arrays(_fn)(input_native)
    assert isinstance(output, ivy.Array)
    assert ivy.as_ivy_dtype(input_native.dtype) == str(output.dtype)
    assert ivy.all(input_native == output.data)

    # check for frontend array
    input_frontend = Tensor(x[0])
    input_frontend.ivy_array = input_ivy
    output = inputs_to_ivy_arrays(_fn)(input_frontend)
    assert isinstance(output, ivy.Array)
    assert str(input_frontend.dtype) == str(output.dtype)
    assert ivy.all(input_frontend.ivy_array == output)


@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
)
def test_outputs_to_frontend_arrays(dtype_and_x):
    x_dtype, x = dtype_and_x

    # check for ivy array
    input_ivy = ivy.array(x[0], dtype=x_dtype[0])
    output = outputs_to_frontend_arrays(_fn)(input_ivy)
    assert isinstance(output, Tensor)
    assert str(input_ivy.dtype) == str(output.dtype)
    assert ivy.all(input_ivy == output.ivy_array)


@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
)
def test_to_ivy_arrays_and_back(dtype_and_x):
    x_dtype, x = dtype_and_x

    # check for ivy array
    input_ivy = ivy.array(x[0], dtype=x_dtype[0])
    output = to_ivy_arrays_and_back(_fn)(input_ivy)
    assert isinstance(output, Tensor)
    assert str(input_ivy.dtype) == str(output.dtype)
    assert ivy.all(input_ivy == output.ivy_array)

    # check for native array
    input_native = ivy.native_array(input_ivy)
    output = to_ivy_arrays_and_back(_fn)(input_native)
    assert isinstance(output, Tensor)
    assert ivy.as_ivy_dtype(input_native.dtype) == str(output.dtype)
    assert ivy.all(input_native == output.ivy_array.data)

    # check for frontend array
    input_frontend = Tensor(x[0])
    input_frontend.ivy_array = input_ivy
    output = to_ivy_arrays_and_back(_fn)(input_frontend)
    assert isinstance(output, Tensor)
    assert input_frontend.dtype == output.dtype
    assert ivy.all(input_frontend.ivy_array == output.ivy_array)
