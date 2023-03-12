# global
from hypothesis import given, strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy.functional.frontends.tensorflow.func_wrapper import (
    inputs_to_ivy_arrays,
    outputs_to_frontend_arrays,
    to_ivy_arrays_and_back,
    handle_tf_dtype,
)
from ivy.functional.frontends.tensorflow.tensor import EagerTensor
import ivy.functional.frontends.tensorflow as tf_frontend
import ivy.functional.frontends.numpy as np_frontend


def _fn(x=None, dtype=None):
    if ivy.exists(dtype):
        return dtype
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
    assert ivy.all(input_native == output.data)

    # check for frontend array
    input_frontend = EagerTensor(x[0])
    output = inputs_to_ivy_arrays(_fn)(input_frontend)
    assert isinstance(output, ivy.Array)
    assert input_frontend.dtype.ivy_dtype == output.dtype
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
    output = outputs_to_frontend_arrays(_fn)(input_ivy)
    assert isinstance(output, EagerTensor)
    assert input_ivy.dtype == output.dtype.ivy_dtype
    assert ivy.all(input_ivy == output.ivy_array)


@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False)
    ),
)
def test_to_ivy_arrays_and_back(dtype_and_x):
    x_dtype, x = dtype_and_x

    # check for ivy array
    input_ivy = ivy.array(x[0], dtype=x_dtype[0])
    output = to_ivy_arrays_and_back(_fn)(input_ivy)
    assert isinstance(output, EagerTensor)
    assert input_ivy.dtype == output.dtype.ivy_dtype
    assert ivy.all(input_ivy == output.ivy_array)

    # check for native array
    input_native = ivy.native_array(input_ivy)
    output = to_ivy_arrays_and_back(_fn)(input_native)
    assert isinstance(output, EagerTensor)
    assert ivy.as_ivy_dtype(input_native.dtype) == output.dtype.ivy_dtype
    assert ivy.all(input_native == output.ivy_array.data)

    # check for frontend array
    input_frontend = EagerTensor(x[0])
    output = to_ivy_arrays_and_back(_fn)(input_frontend)
    assert isinstance(output, EagerTensor)
    assert input_frontend.dtype == output.dtype
    assert ivy.all(input_frontend.ivy_array == output.ivy_array)


@st.composite
def _dtype_helper(draw):
    return draw(
        st.sampled_from(
            [
                draw(helpers.get_dtypes("valid", prune_function=False, full=False))[0],
                ivy.as_native_dtype(
                    draw(helpers.get_dtypes("valid", prune_function=False, full=False))[
                        0
                    ]
                ),
                draw(
                    st.sampled_from(list(tf_frontend.tensorflow_enum_to_type.values()))
                ),
                draw(st.sampled_from(list(tf_frontend.tensorflow_enum_to_type.keys()))),
                np_frontend.dtype(
                    draw(helpers.get_dtypes("valid", prune_function=False, full=False))[
                        0
                    ]
                ),
                draw(st.sampled_from(list(np_frontend.numpy_scalar_to_dtype.keys()))),
            ]
        )
    )


@given(
    dtype=_dtype_helper(),
)
def test_handle_tf_dtype(dtype):
    ret_dtype = handle_tf_dtype(_fn)(dtype=dtype)
    assert isinstance(ret_dtype, ivy.Dtype)
