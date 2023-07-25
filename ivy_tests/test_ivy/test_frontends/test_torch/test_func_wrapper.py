# global
from hypothesis import given, strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy.functional.frontends.torch.func_wrapper import (
    inputs_to_ivy_arrays,
    outputs_to_frontend_arrays,
    to_ivy_arrays_and_back,
    numpy_to_torch_style_args,
)
from ivy.functional.frontends.torch.tensor import Tensor
import ivy.functional.frontends.torch as torch_frontend


def _fn(*args, dtype=None, check_default=False):
    if (
        check_default
        and all([not (ivy.is_array(i) or hasattr(i, "ivy_array")) for i in args])
        and not ivy.exists(dtype)
    ):
        ivy.utils.assertions.check_equal(
            ivy.default_float_dtype(),
            torch_frontend.get_default_dtype(),
            as_array=False,
        )
        ivy.utils.assertions.check_equal(
            ivy.default_int_dtype(), "int64", as_array=False
        )
    return args[0]


@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False)
    ).filter(lambda x: "bfloat16" not in x[0])
)
def test_torch_inputs_to_ivy_arrays(dtype_and_x, backend_fw):
    x_dtype, x = dtype_and_x

    ivy.set_backend(backend=backend_fw)

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

    ivy.previous_backend()


@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False)
    ).filter(lambda x: "bfloat16" not in x[0]),
    dtype=helpers.get_dtypes("valid", none=True, full=False, prune_function=False),
)
def test_torch_outputs_to_frontend_arrays(dtype_and_x, dtype, backend_fw):
    x_dtype, x = dtype_and_x

    ivy.set_backend(backend_fw)

    # check for ivy array
    input_ivy = ivy.array(x[0], dtype=x_dtype[0])
    if not len(input_ivy.shape):
        scalar_input_ivy = ivy.to_scalar(input_ivy)
        outputs_to_frontend_arrays(_fn)(
            scalar_input_ivy, scalar_input_ivy, check_default=True, dtype=dtype
        )
        outputs_to_frontend_arrays(_fn)(
            scalar_input_ivy, input_ivy, check_default=True, dtype=dtype
        )
    output = outputs_to_frontend_arrays(_fn)(input_ivy, check_default=True, dtype=dtype)
    assert isinstance(output, Tensor)
    assert str(input_ivy.dtype) == str(output.dtype)
    assert ivy.all(input_ivy == output.ivy_array)

    assert ivy.default_float_dtype_stack == ivy.default_int_dtype_stack == []

    ivy.previous_backend()


@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False)
    ).filter(lambda x: "bfloat16" not in x[0]),
    dtype=helpers.get_dtypes("valid", none=True, full=False, prune_function=False),
)
def test_torch_to_ivy_arrays_and_back(dtype_and_x, dtype, backend_fw):
    x_dtype, x = dtype_and_x

    ivy.set_backend(backend_fw)

    # check for ivy array
    input_ivy = ivy.array(x[0], dtype=x_dtype[0])
    if not len(input_ivy.shape):
        scalar_input_ivy = ivy.to_scalar(input_ivy)
        to_ivy_arrays_and_back(_fn)(
            scalar_input_ivy, scalar_input_ivy, check_default=True, dtype=dtype
        )
        to_ivy_arrays_and_back(_fn)(
            scalar_input_ivy, input_ivy, check_default=True, dtype=dtype
        )
    output = to_ivy_arrays_and_back(_fn)(input_ivy, check_default=True, dtype=dtype)
    assert isinstance(output, Tensor)
    assert str(input_ivy.dtype) == str(output.dtype)
    assert ivy.all(input_ivy == output.ivy_array)

    # check for native array
    input_native = ivy.native_array(input_ivy)
    if not len(input_native.shape):
        scalar_input_native = ivy.to_scalar(input_native)
        to_ivy_arrays_and_back(_fn)(
            scalar_input_native, scalar_input_native, check_default=True, dtype=dtype
        )
        to_ivy_arrays_and_back(_fn)(
            scalar_input_native, input_native, check_default=True, dtype=dtype
        )
    output = to_ivy_arrays_and_back(_fn)(input_native, check_default=True, dtype=dtype)
    assert isinstance(output, Tensor)
    assert ivy.as_ivy_dtype(input_native.dtype) == str(output.dtype)
    assert ivy.all(input_native == output.ivy_array.data)

    # check for frontend array
    input_frontend = Tensor(x[0])
    input_frontend.ivy_array = input_ivy
    if not len(input_frontend.shape):
        scalar_input_front = inputs_to_ivy_arrays(ivy.to_scalar)(input_frontend)
        to_ivy_arrays_and_back(_fn)(
            scalar_input_front, scalar_input_front, check_default=True, dtype=dtype
        )
        to_ivy_arrays_and_back(_fn)(
            scalar_input_front, input_frontend, check_default=True, dtype=dtype
        )
    output = to_ivy_arrays_and_back(_fn)(
        input_frontend, check_default=True, dtype=dtype
    )
    assert isinstance(output, Tensor)
    assert input_frontend.dtype == output.dtype
    assert ivy.all(input_frontend.ivy_array == output.ivy_array)

    assert ivy.default_float_dtype_stack == ivy.default_int_dtype_stack == []

    ivy.previous_backend()


@numpy_to_torch_style_args
def mocked_func(dim=None, keepdim=None, input=None, other=None):
    return dim, keepdim, input, other


@given(
    dim=st.integers(),
    keepdim=st.booleans(),
    input=st.lists(st.integers()),
    other=st.integers(),
)
def test_torch_numpy_to_torch_style_args(dim, keepdim, input, other):
    # PyTorch-style keyword arguments
    assert (dim, keepdim, input, other) == mocked_func(
        dim=dim, keepdim=keepdim, input=input, other=other
    )

    # NumPy-style keyword arguments
    assert (dim, keepdim, input, other) == mocked_func(
        axis=dim, keepdims=keepdim, x=input, x2=other
    )

    # Mixed-style keyword arguments
    assert (dim, keepdim, input, other) == mocked_func(
        axis=dim, keepdim=keepdim, input=input, x2=other
    )
