# global
from hypothesis import given

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy.functional.frontends.jax.func_wrapper import (
    inputs_to_ivy_arrays,
    outputs_to_frontend_arrays,
    to_ivy_arrays_and_back,
)
from ivy.functional.frontends.jax.devicearray import DeviceArray
import ivy.functional.frontends.jax as jax_frontend


def _fn(x, check_default=False):
    if check_default and jax_frontend.config.jax_enable_x64:
        ivy.utils.assertions.check_equal(ivy.default_float_dtype(), "float64")
        ivy.utils.assertions.check_equal(ivy.default_int_dtype(), "int64")
    return x


@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False)
    ),
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
    assert ivy.as_ivy_dtype(input_native.dtype) == output.dtype
    assert ivy.all(ivy.equal(input_native, output.data))

    # check for frontend array
    input_frontend = DeviceArray(x[0])
    output = inputs_to_ivy_arrays(_fn)(input_frontend)
    assert isinstance(output, ivy.Array)
    assert input_frontend.dtype == output.dtype
    assert ivy.all(input_frontend.ivy_array == output)


@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False)
    ),
)
def test_outputs_to_frontend_arrays(dtype_and_x):
    x_dtype, x = dtype_and_x

    # check for ivy array
    input_ivy = ivy.array(x[0], dtype=x_dtype[0])
    output = outputs_to_frontend_arrays(_fn)(input_ivy, check_default=True)
    assert isinstance(output, DeviceArray)
    assert input_ivy.dtype == output.dtype
    assert ivy.all(input_ivy == output.ivy_array)

    assert ivy.default_float_dtype_stack == ivy.default_int_dtype_stack == []


@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False)
    ),
)
def test_to_ivy_arrays_and_back(dtype_and_x):
    x_dtype, x = dtype_and_x

    # check for ivy array
    input_ivy = ivy.array(x[0], dtype=x_dtype[0])
    output = to_ivy_arrays_and_back(_fn)(input_ivy, check_default=True)
    assert isinstance(output, DeviceArray)
    assert input_ivy.dtype == output.dtype
    assert ivy.all(input_ivy == output.ivy_array)

    # check for native array
    input_native = ivy.native_array(input_ivy)
    output = to_ivy_arrays_and_back(_fn)(input_native, check_default=True)
    assert isinstance(output, DeviceArray)
    assert ivy.as_ivy_dtype(input_native.dtype) == output.dtype
    assert ivy.all(ivy.equal(input_native, output.ivy_array.data))

    # check for frontend array
    input_frontend = DeviceArray(x[0])
    output = to_ivy_arrays_and_back(_fn)(input_frontend, check_default=True)
    assert isinstance(output, DeviceArray)
    assert str(input_frontend.dtype) == str(output.dtype)
    assert ivy.all(input_frontend.ivy_array == output.ivy_array)

    assert ivy.default_float_dtype_stack == ivy.default_int_dtype_stack == []
