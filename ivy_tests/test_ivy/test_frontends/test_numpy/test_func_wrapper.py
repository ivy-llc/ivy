# global
from hypothesis import given, strategies as st
import platform

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy.functional.frontends.numpy.func_wrapper import (
    inputs_to_ivy_arrays,
    outputs_to_numpy_arrays,
    to_ivy_arrays_and_back,
    handle_numpy_dtype,
    from_zero_dim_arrays_to_scalar,
)
from ivy.functional.frontends.numpy.ndarray import ndarray
import ivy.functional.frontends.numpy as np_frontend


def _fn(x=None, check_default=False, dtype=None):
    if check_default:
        ivy.assertions.check_equal(ivy.default_float_dtype(), "float64")
        if platform.system() != "Windows":
            ivy.assertions.check_equal(ivy.default_int_dtype(), "int64")
        else:
            ivy.assertions.check_equal(ivy.default_int_dtype(), "int32")
    if ivy.exists(dtype):
        return dtype
    return x


@given(
    dtype_x_shape=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        ret_shape=True,
    ),
)
def test_inputs_to_ivy_arrays(dtype_x_shape):
    x_dtype, x, shape = dtype_x_shape

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
    input_frontend = ndarray(shape)
    input_frontend.ivy_array = input_ivy
    output = inputs_to_ivy_arrays(_fn)(input_frontend)
    assert isinstance(output, ivy.Array)
    assert input_frontend.ivy_array.dtype == str(output.dtype)
    assert ivy.all(input_frontend.ivy_array == output)


@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
)
def test_outputs_to_numpy_arrays(dtype_and_x):
    x_dtype, x = dtype_and_x

    # check for ivy array
    input_ivy = ivy.array(x[0], dtype=x_dtype[0])
    output = outputs_to_numpy_arrays(_fn)(input_ivy, check_default=True)
    assert isinstance(output, ndarray)
    assert input_ivy.dtype == output.ivy_array.dtype
    assert ivy.all(input_ivy == output.ivy_array)

    assert ivy.default_float_dtype_stack == ivy.default_int_dtype_stack == []


@given(
    dtype_x_shape=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        ret_shape=True,
    ),
)
def test_to_ivy_arrays_and_back(dtype_x_shape):
    x_dtype, x, shape = dtype_x_shape

    # check for ivy array
    input_ivy = ivy.array(x[0], dtype=x_dtype[0])
    output = to_ivy_arrays_and_back(_fn)(input_ivy, check_default=True)
    assert isinstance(output, ndarray)
    assert input_ivy.dtype == output.ivy_array.dtype
    assert ivy.all(input_ivy == output.ivy_array)

    # check for native array
    input_native = ivy.native_array(input_ivy)
    output = to_ivy_arrays_and_back(_fn)(input_native, check_default=True)
    assert isinstance(output, ndarray)
    assert ivy.as_ivy_dtype(input_native.dtype) == output.ivy_array.dtype
    assert ivy.all(input_native == output.ivy_array.data)

    # check for frontend array
    input_frontend = ndarray(shape)
    input_frontend.ivy_array = input_ivy
    output = to_ivy_arrays_and_back(_fn)(input_frontend, check_default=True)
    assert isinstance(output, ndarray)
    assert input_frontend.ivy_array.dtype == output.ivy_array.dtype
    assert ivy.all(input_frontend.ivy_array == output.ivy_array)

    assert ivy.default_float_dtype_stack == ivy.default_int_dtype_stack == []


@st.composite
def _zero_dim_to_scalar_helper(draw):
    dtype = draw(
        helpers.get_dtypes("valid", full=False).filter(lambda x: "bfloat16" not in x)
    )[0]
    shape = draw(helpers.get_shape())
    return draw(
        st.one_of(
            helpers.array_values(shape=shape, dtype=dtype),
            st.lists(helpers.array_values(shape=shape, dtype=dtype), min_size=1).map(
                tuple
            ),
        )
    )


def _zero_dim_to_scalar_checks(x, ret_x):
    if len(x.shape) > 0:
        assert ivy.all(ivy.array(ret_x) == ivy.array(x))
    else:
        assert issubclass(type(ret_x), np_frontend.generic)
        assert ret_x.ivy_array == ivy.array(x)


@given(x=_zero_dim_to_scalar_helper())
def test_from_zero_dim_arrays_to_scalar(x):
    ret_x = from_zero_dim_arrays_to_scalar(_fn)(x=x)
    if isinstance(x, tuple):
        assert isinstance(ret_x, tuple)
        for x_i, ret_x_i in zip(x, ret_x):
            _zero_dim_to_scalar_checks(x_i, ret_x_i)
    else:
        _zero_dim_to_scalar_checks(x, ret_x)


@st.composite
def _dtype_helper(draw):
    return draw(
        st.sampled_from(
            [
                draw(st.sampled_from([int, float, bool])),
                ivy.as_native_dtype(draw(helpers.get_dtypes("valid", full=False))[0]),
                np_frontend.dtype(draw(helpers.get_dtypes("valid", full=False))[0]),
                draw(st.sampled_from(list(np_frontend.numpy_scalar_to_dtype.keys()))),
                draw(st.sampled_from(list(np_frontend.numpy_str_to_type_table.keys()))),
            ]
        )
    )


@given(
    dtype=_dtype_helper(),
)
def test_handle_numpy_dtype(dtype):
    ret_dtype = handle_numpy_dtype(_fn)(dtype=dtype)
    assert isinstance(ret_dtype, ivy.Dtype)
