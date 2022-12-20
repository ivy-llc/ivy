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
    input = Tensor(x[0])
    input.ivy_array = ivy.array(x[0], dtype=x_dtype[0])
    output = inputs_to_ivy_arrays(_fn)(input)
    assert isinstance(output, ivy.Array)
    assert str(input.dtype) == str(output.dtype)


@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
)
def test_outputs_to_frontend_arrays(dtype_and_x):
    x_dtype, x = dtype_and_x
    input = ivy.array(x[0], dtype=x_dtype[0])
    output = outputs_to_frontend_arrays(_fn)(input)
    assert isinstance(output, Tensor)
    assert str(input.dtype) == str(output.dtype)


@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
)
def test_to_ivy_arrays_and_back(dtype_and_x):
    x_dtype, x = dtype_and_x
    input = Tensor(x[0])
    input.ivy_array = ivy.array(x[0], dtype=x_dtype[0])
    output = to_ivy_arrays_and_back(_fn)(input)
    assert isinstance(output, Tensor)
    assert input.dtype == output.dtype
