# global
from hypothesis import given, strategies as st
import torch

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy.functional.frontends.torch.func_wrapper import (
    inputs_to_ivy_arrays,
    outputs_to_frontend_arrays,
    to_ivy_arrays_and_back,
    numpy_to_torch_style_args,
    handle_gradients,
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
    dtype_and_xs=helpers.dtype_and_values(
        min_value=-1e5,
        max_value=1e5,
        available_dtypes=("float32", "float64"),
    ),
)
def test_handle_gradients(dtype_and_xs):
    if ivy.current_backend_str() == "paddle" or ivy.current_backend_str() == "numpy":
        return

    dtypes, xs = dtype_and_xs
    fn = lambda x: 2 * x

    # Test when requires_grad = False
    x = torch_frontend.tensor(xs[0], dtype=dtypes[0])

    output_fn = fn(x)
    output_wrapped = handle_gradients(fn)(x)

    assert ivy.all(output_fn.ivy_array == output_wrapped.ivy_array)
    assert output_wrapped.jac_fn is None
    assert output_wrapped.func_inputs is None
    assert not output_wrapped.requires_grad

    # Test when requires_grad = True
    # Test requires_grad is set properly
    x = torch_frontend.tensor(xs[0], requires_grad=True)
    assert x.requires_grad

    output_wrapped = handle_gradients(fn)(x)
    assert output_wrapped.requires_grad

    # Test function inputs are stored
    for stored, gt in zip(output_wrapped.func_inputs, [x]):
        assert ivy.all(stored[0].ivy_array == gt.ivy_array)

    # Test jac_fn is stored
    x_native = torch.tensor(xs[0], requires_grad=True)
    output_native = fn(x_native)
    grads_gt = torch.autograd.grad(
        output_native, x_native, grad_outputs=torch.ones_like(output_native)
    )

    axis = list(range(len(output_wrapped.shape)))
    grads = output_wrapped.jac_fn([[x], {}])[0][0].sum(dim=axis)

    ivy.set_backend("torch")
    grads_flat_np_gt = helpers.flatten_and_to_np(ret=ivy.to_ivy(grads_gt[0]))
    ivy.previous_backend()
    grads_flat_np = helpers.flatten_frontend_to_np(ret=grads)
    for grads_flat, grads_flat_gt in zip(grads_flat_np, grads_flat_np_gt):
        assert grads_flat.shape == grads_flat_gt.shape
        helpers.value_test(ret_np_flat=grads_flat, ret_np_from_gt_flat=grads_flat_gt)


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
