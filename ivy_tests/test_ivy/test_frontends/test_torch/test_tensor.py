# global
import pytest
from types import SimpleNamespace
import numpy as np

from ivy_tests.test_ivy.test_frontends.test_torch.test_comparison_ops import (
    _topk_helper,
)
from ivy_tests.test_ivy.test_frontends.test_torch.test_creation_ops import (
    _as_strided_helper,
)
from ivy_tests.test_ivy.test_frontends.test_torch.test_indexing_slicing_joining_mutating_ops import (  # noqa: E501
    _dtype_input_dim_start_length,
)
from ivy_tests.test_ivy.test_frontends.test_torch.test_reduction_ops import (
    _get_axis_and_p,
)

try:
    import torch
except ImportError:
    torch = SimpleNamespace()

import ivy
from hypothesis import strategies as st, given, assume

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.test_frontends.test_torch.test_blas_and_lapack_ops import (
    _get_dtype_and_3dbatch_matrices,
    _get_dtype_input_and_matrices,
    _get_dtype_input_and_mat_vec,
)
from ivy.functional.frontends.torch import Tensor
from ivy_tests.test_ivy.helpers import handle_frontend_method
from ivy_tests.test_ivy.test_functional.test_core.test_searching import (
    _broadcastable_trio,
)
from ivy_tests.test_ivy.test_functional.test_core.test_manipulation import (  # noqa
    _get_splits,
)
from ivy_tests.test_ivy.test_frontends.test_torch.test_miscellaneous_ops import (  # noqa
    dtype_value1_value2_axis,
)
from ivy_tests.test_ivy.test_frontends.test_torch.test_linalg import (  # noqa
    _get_dtype_and_matrix,
)
from ivy_tests.test_ivy.test_functional.test_core.test_statistical import (
    _get_castable_dtype,
    _statistical_dtype_values,
)

CLASS_TREE = "ivy.functional.frontends.torch.Tensor"


# Helper functions
@st.composite
def _dtypes(draw):
    return draw(
        st.shared(
            helpers.list_of_size(
                x=st.sampled_from(
                    draw(helpers.get_dtypes("numeric", prune_function=False))
                ),
                size=1,
            ),
            key="dtype",
        )
    )


@st.composite
def _requires_grad(draw):
    dtype = draw(_dtypes())[0]
    if ivy.is_int_dtype(dtype) or ivy.is_uint_dtype(dtype):
        return draw(st.just(False))
    return draw(st.booleans())


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False)
    ).filter(lambda x: "bfloat16" not in x[0]),
)
def test_torch_tensor_property_ivy_array(
    dtype_x,
):
    _, data = dtype_x
    x = Tensor(data[0])
    x.ivy_array = data[0]
    ret = helpers.flatten_and_to_np(ret=x.ivy_array.data)
    ret_gt = helpers.flatten_and_to_np(ret=data[0])
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend="torch",
    )


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False)
    ).filter(lambda x: "bfloat16" not in x[0]),
)
def test_torch_tensor_property_device(
    dtype_x,
):
    _, data = dtype_x
    x = Tensor(data[0])
    x.ivy_array = data[0]
    ivy.utils.assertions.check_equal(
        x.device, ivy.dev(ivy.array(data[0])), as_array=False
    )


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False)
    ).filter(lambda x: "bfloat16" not in x[0]),
)
def test_torch_tensor_property_dtype(
    dtype_x,
):
    dtype, data = dtype_x
    x = Tensor(data[0])
    x.ivy_array = data[0]
    ivy.utils.assertions.check_equal(x.dtype, dtype[0], as_array=False)


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False),
        ret_shape=True,
    ).filter(lambda x: "bfloat16" not in x[0]),
)
def test_torch_tensor_property_shape(dtype_x):
    dtype, data, shape = dtype_x
    x = Tensor(data[0])
    ivy.utils.assertions.check_equal(
        x.ivy_array.shape, ivy.Shape(shape), as_array=False
    )


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("complex", prune_function=False)
    ).filter(lambda x: "bfloat16" not in x[0]),
)
def test_torch_tensor_property_real(
    dtype_x,
):
    _, data = dtype_x
    x = Tensor(data[0])
    x.ivy_array = data[0]
    ivy.utils.assertions.check_equal(x.real, ivy.real(data[0]))


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("complex", prune_function=False)
    ),
)
def test_torch_tensor_property_imag(
    dtype_x,
):
    _, data = dtype_x
    x = Tensor(data[0])
    x.ivy_array = data[0]
    ivy.utils.assertions.check_equal(x.imag, ivy.imag(data[0]))


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False),
        ret_shape=True,
    ).filter(lambda x: "bfloat16" not in x[0]),
)
def test_torch_tensor_property_ndim(
    dtype_x,
):
    dtype, data, shape = dtype_x
    x = Tensor(data[0])
    ivy.utils.assertions.check_equal(x.ndim, data[0].ndim, as_array=False)


def test_torch_tensor_property_grad():
    x = Tensor(ivy.array([1.0, 2.0, 3.0]))
    grads = ivy.array([1.0, 2.0, 3.0])
    x._grads = grads
    assert ivy.array_equal(x.grad, grads)


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False),
    ),
    requires_grad=st.booleans(),
)
def test_torch_tensor_property_requires_grad(
    dtype_x,
    requires_grad,
):
    _, data = dtype_x
    x = Tensor(data[0], requires_grad=requires_grad)
    ivy.utils.assertions.check_equal(x.requires_grad, requires_grad, as_array=False)
    x.requires_grad = not requires_grad
    ivy.utils.assertions.check_equal(x.requires_grad, not requires_grad, as_array=False)


@given(
    requires_grad=st.booleans(),
)
def test_torch_tensor_property_is_leaf(
    requires_grad,
):
    x = Tensor(ivy.array([3.0]), requires_grad=requires_grad)
    ivy.utils.assertions.check_equal(x.is_leaf, True, as_array=False)
    y = x.pow(2)
    ivy.utils.assertions.check_equal(y.is_leaf, not requires_grad, as_array=False)
    z = y.detach()
    ivy.utils.assertions.check_equal(z.is_leaf, True, as_array=False)


def test_torch_tensor_property_grad_fn():
    x = Tensor(ivy.array([3.0]), requires_grad=True)
    ivy.utils.assertions.check_equal(x.grad_fn, None, as_array=False)
    y = x.pow(2)
    ivy.utils.assertions.check_equal(y.grad_fn, "PowBackward", as_array=False)
    ivy.utils.assertions.check_equal(
        y.grad_fn.next_functions[0], "AccumulateGrad", as_array=False
    )
    z = y.detach()
    ivy.utils.assertions.check_equal(z.grad_fn, None, as_array=False)


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False),
    ),
    requires_grad=st.booleans(),
)
def test_torch_tensor_requires_grad_(
    dtype_x,
    requires_grad,
):
    _, data = dtype_x
    x = Tensor(data[0])
    assert not x._requires_grad
    x.requires_grad_()
    assert x._requires_grad
    x.requires_grad_(requires_grad)
    assert x._requires_grad == requires_grad


# chunk
@pytest.mark.skip("Testing takes a lot of time")
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="chunk",
    dtype_x_dim=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        min_value=-1e04,
        max_value=1e04,
        force_int_axis=True,
        valid_axis=True,
    ),
    chunks=st.integers(
        min_value=1,
        max_value=5,
    ),
)
def test_torch_instance_chunk(
    dtype_x_dim,
    chunks,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x, dim = dtype_x_dim
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "chunks": chunks,
            "dim": dim,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# any
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="any",
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        min_value=-1e04,
        max_value=1e04,
        valid_axis=True,
        force_int_axis=True,
    ),
    keepdim=st.booleans(),
)
def test_torch_instance_any(
    dtype_input_axis,
    keepdim,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x, axis = dtype_input_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dim": axis,
            "keepdim": keepdim,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# all
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="all",
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        min_value=-1e04,
        max_value=1e04,
        valid_axis=True,
        force_int_axis=True,
    ),
    keepdim=st.booleans(),
)
def test_torch_instance_all(
    dtype_input_axis,
    keepdim,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x, axis = dtype_input_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dim": axis,
            "keepdim": keepdim,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# add
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="add",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
    alpha=st.floats(min_value=-1e04, max_value=1e04, allow_infinity=False),
)
def test_torch_instance_add(
    dtype_and_x,
    alpha,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
            "alpha": alpha,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        atol_=1e-02,
        on_device=on_device,
    )


# addmm
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="addmm",
    dtype_and_matrices=_get_dtype_input_and_matrices(with_input=True),
    beta=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
    alpha=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
)
def test_torch_instance_addmm(
    dtype_and_matrices,
    beta,
    alpha,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x, mat1, mat2 = dtype_and_matrices
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x,
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "mat1": mat1,
            "mat2": mat2,
            "beta": beta,
            "alpha": alpha,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        atol_=1e-02,
        on_device=on_device,
    )


# addmm_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="addmm_",
    dtype_and_matrices=_get_dtype_input_and_matrices(with_input=True),
    beta=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
    alpha=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
)
def test_torch_instance_addmm_(
    dtype_and_matrices,
    beta,
    alpha,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x, mat1, mat2 = dtype_and_matrices
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x,
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "mat1": mat1,
            "mat2": mat2,
            "beta": beta,
            "alpha": alpha,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        atol_=1e-02,
        on_device=on_device,
    )


# addmv
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="addmv",
    dtype_and_matrices=_get_dtype_input_and_mat_vec(with_input=True),
    beta=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
    alpha=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
)
def test_torch_instance_addmv(
    dtype_and_matrices,
    beta,
    alpha,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x, mat, vec = dtype_and_matrices
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x,
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "mat": mat,
            "vec": vec,
            "beta": beta,
            "alpha": alpha,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        atol_=1e-02,
        on_device=on_device,
    )


# addbmm
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="addbmm",
    dtype_and_matrices=_get_dtype_and_3dbatch_matrices(with_input=True),
    beta=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
    alpha=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
)
def test_torch_instance_addbmm(
    dtype_and_matrices,
    beta,
    alpha,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x, batch1, batch2 = dtype_and_matrices
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x,
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "batch1": batch1,
            "batch2": batch2,
            "beta": beta,
            "alpha": alpha,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        atol_=1e-02,
        on_device=on_device,
    )


# addbmm_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="addbmm_",
    dtype_and_matrices=_get_dtype_and_3dbatch_matrices(with_input=True),
    beta=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
    alpha=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
)
def test_torch_instance_addbmm_(
    dtype_and_matrices,
    beta,
    alpha,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x, batch1, batch2 = dtype_and_matrices
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x,
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "batch1": batch1,
            "batch2": batch2,
            "beta": beta,
            "alpha": alpha,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        atol_=1e-02,
        on_device=on_device,
    )


# sub
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="sub",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
    alpha=st.floats(min_value=-1e04, max_value=1e04, allow_infinity=False),
)
def test_torch_instance_sub(
    dtype_and_x,
    alpha,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
            "alpha": alpha,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        atol_=1e-02,
        on_device=on_device,
    )


# new_ones
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="new_ones",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    size=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtypes=_dtypes(),
    requires_grad=_requires_grad(),
)
def test_torch_instance_new_ones(
    dtype_and_x,
    size,
    dtypes,
    requires_grad,
    on_device,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=dtypes,
        method_all_as_kwargs_np={
            "size": size,
            "dtype": dtypes[0],
            "requires_grad": requires_grad,
            "device": on_device,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# new_zeros
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="new_zeros",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
    size=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtypes=_dtypes(),
    requires_grad=_requires_grad(),
)
def test_torch_instance_new_zeros(
    dtype_and_x,
    size,
    dtypes,
    requires_grad,
    on_device,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=dtypes,
        method_all_as_kwargs_np={
            "size": size,
            "dtype": dtypes[0],
            "requires_grad": requires_grad,
            "device": on_device,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="reshape",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(), key="value_shape"),
    ),
    shape=helpers.reshape_shapes(
        shape=st.shared(helpers.get_shape(), key="value_shape")
    ),
    unpack_shape=st.booleans(),
)
def test_torch_instance_reshape(
    dtype_x,
    shape,
    unpack_shape,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_x
    shape = {
        "shape": shape,
    }
    if unpack_shape:
        method_flags.num_positional_args = len(shape["shape"]) + 1
        i = 0
        for x_ in shape["shape"]:
            shape["x{}".format(i)] = x_
            i += 1
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np=shape,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# reshape_as
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="reshape_as",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"), num_arrays=2
    ),
)
def test_torch_instance_reshape_as(
    dtype_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# sin
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="sin",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_sin(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# arcsin
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="arcsin",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_arcsin(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# sum
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="sum",
    dtype_x_dim=_get_castable_dtype(
        min_value=-1e04,
        max_value=1e04,
    ),
    keepdim=st.booleans(),
)
def test_torch_instance_sum(
    dtype_x_dim,
    keepdim,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x, dim, castable_dtype = dtype_x_dim
    if method_flags.as_variable:
        castable_dtype = input_dtype
    input_dtype = [input_dtype]
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dim": dim,
            "keepdim": keepdim,
            "dtype": castable_dtype,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# atan
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="atan",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_atan(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# atan2
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="atan2",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
    ),
)
def test_torch_instance_atan2(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# sin_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="sin_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_sin_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# cos
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="cos",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_cos(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# cos_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="cos_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_cos_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": list(x[0]) if type(x[0]) == int else x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# sinh
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="sinh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_sinh(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# sinh_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="sinh_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_sinh_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# cosh
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="cosh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_cosh(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# cosh_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="cosh_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_cosh_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
        rtol_=1e-2,
        atol_=1e-2,
    )


# view
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="view",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(), key="value_shape"),
    ),
    shape=helpers.reshape_shapes(
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape")
    ),
)
def test_torch_instance_view(
    dtype_x,
    shape,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "size": shape,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="float",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_torch_instance_float(
    dtype_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# asinh
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="asinh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_asinh(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        rtol_=1e-2,
        atol_=1e-2,
        on_device=on_device,
    )


# asinh_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="asinh_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_asinh_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        rtol_=1e-2,
        atol_=1e-2,
        on_device=on_device,
    )


# tan
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="tan",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_tan(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# tanh
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="tanh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_tanh(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# tanh_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="tanh_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_tanh_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# asin
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="asin",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_asin(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# amax
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="amax",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        valid_axis=True,
        force_int_axis=True,
    ),
    keepdim=st.booleans(),
)
def test_torch_instance_amax(
    dtype_x_axis,
    keepdim,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dim": axis,
            "keepdim": keepdim,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# abs
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="abs",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_instance_abs(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# abs_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="abs_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_instance_abs_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# amin
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="amin",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        valid_axis=True,
        force_int_axis=True,
    ),
    keepdim=st.booleans(),
)
def test_torch_instance_amin(
    dtype_x_axis,
    keepdim,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dim": axis,
            "keepdim": keepdim,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# aminmax
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="aminmax",
    dtype_input_axis=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
)
def test_torch_instance_aminmax(
    dtype_input_axis,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_input_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# bernoulli
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="bernoulli",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    test_with_out=st.just(True),
)
def test_torch_instance_bernoulli(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "input": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"generator": x[1], "out": x[2]},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
    )


# contiguous
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="contiguous",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_contiguous(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# log
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="log",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_log(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# log_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="log_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_log_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# log2
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="log2",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_log2(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __bool__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__bool__",
    dtype_and_x=helpers.dtype_and_values(
        max_dim_size=1,
        min_value=-1e04,
        max_value=1e04,
    ),
)
def test_torch_special_bool(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __add__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__add__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch_special_add(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# arcsinh
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="arcsinh",
    dtype_and_x=helpers.dtype_and_values(
        min_value=-1.0,
        max_value=1.0,
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_instance_arcsinh(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __long__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__long__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch_special_long(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __radd__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__radd__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch_special_radd(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __sub__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__sub__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch_special_sub(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __mul__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__mul__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch_special_mul(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@st.composite
def _get_dtype_and_multiplicative_matrices(draw):
    return draw(
        st.one_of(
            _get_dtype_input_and_matrices(),
            _get_dtype_and_3dbatch_matrices(),
        )
    )


# __matmul__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__matmul__",
    dtype_tensor1_tensor2=_get_dtype_and_multiplicative_matrices(),
)
def test_torch_special_matmul(
    dtype_tensor1_tensor2,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    dtype, tensor1, tensor2 = dtype_tensor1_tensor2
    helpers.test_frontend_method(
        init_input_dtypes=dtype,
        init_all_as_kwargs_np={
            "data": tensor1,
        },
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={"other": tensor2},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __rsub__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__rsub__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
)
def test_torch_special_rsub(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __rmul__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__rmul__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch_special_rmul(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __truediv__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__truediv__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shared_dtype=True,
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch_special_truediv(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__floordiv__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
    ),
)
def test_torch_special_floordiv(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x[1], 0)))
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
        atol_=1,
    )


# remainder
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="remainder",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        shared_dtype=True,
        num_arrays=2,
    ),
)
def test_torch_remainder(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@st.composite
def _to_helper(draw):
    dtype_x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            num_arrays=2,
            large_abs_safety_factor=3,
        )
    )
    input_dtype, x = dtype_x
    arg = draw(st.sampled_from(["tensor", "dtype", "device"]))
    if arg == "tensor":
        method_num_positional_args = 1
        method_all_as_kwargs_np = {"other": x[1]}
    elif arg == "dtype":
        method_num_positional_args = 1
        dtype = draw(helpers.get_dtypes("valid", full=False))[0]
        method_all_as_kwargs_np = {"dtype": dtype}
    else:
        method_num_positional_args = 0
        device = draw(st.just("cpu"))
        dtype = draw(helpers.get_dtypes("valid", full=False, none=True))[0]
        method_all_as_kwargs_np = {"dtype": dtype, "device": device}
    return input_dtype, x, method_num_positional_args, method_all_as_kwargs_np


# to
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="to",
    args_kwargs=_to_helper(),
)
def test_torch_instance_to(
    args_kwargs,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x, method_num_positional_args, method_all_as_kwargs_np = args_kwargs
    method_flags.num_positional_args = method_num_positional_args
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np=method_all_as_kwargs_np,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# arctan
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="arctan",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_arctan(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# arctan_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="arctan_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_arctan_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# arctan2
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="arctan2",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
    ),
)
def test_torch_instance_arctan2(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# arctan2_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="arctan2_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
    ),
)
def test_torch_instance_arctan2_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# acos
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="acos",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_acos(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# floor
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="floor",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_instance_floor(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# new_tensor
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="new_tensor",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
)
def test_torch_instance_new_tensor(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtype[0]],
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[input_dtype[1]],
        method_all_as_kwargs_np={
            "data": x[1],
            "dtype": input_dtype[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __getitem__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__getitem__",
    dtype_x_index=helpers.dtype_array_query(
        available_dtypes=helpers.get_dtypes("valid"),
        allow_neg_step=False,
    ),
)
def test_torch_instance_getitem(
    dtype_x_index,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x, index = dtype_x_index
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtype[0]],
        init_all_as_kwargs_np={"data": x},
        method_input_dtypes=[*input_dtype[1:]],
        method_all_as_kwargs_np={"query": index},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __setitem__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__setitem__",
    dtypes_x_index_val=helpers.dtype_array_query_val(
        available_dtypes=helpers.get_dtypes("valid"),
        allow_neg_step=False,
    ).filter(lambda x: x[0][0] == x[0][-1]),
)
def test_torch_instance_setitem(
    dtypes_x_index_val,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x, index, val = dtypes_x_index_val
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtype[0]],
        init_all_as_kwargs_np={"data": x},
        method_input_dtypes=[*input_dtype[1:]],
        method_all_as_kwargs_np={"key": index, "value": val},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# view_as
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="view_as",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        shape=st.shared(helpers.get_shape(), key="value_shape"),
        num_arrays=2,
    ),
)
def test_torch_instance_view_as(
    dtype_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# unsqueeze
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="unsqueeze",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(), key="shape"),
    ),
    dim=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="shape"),
        allow_neg=True,
        force_int=True,
    ),
)
def test_torch_instance_unsqueeze(
    dtype_value,
    dim,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dim": dim,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# unsqueeze_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="unsqueeze_",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(), key="shape"),
    ),
    dim=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="shape"),
        allow_neg=True,
        force_int=True,
    ),
)
def test_torch_instance_unsqueeze_(
    dtype_value,
    dim,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dim": dim,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# ravel
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="ravel",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ),
)
def test_torch_instance_ravel(
    dtype_value,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# split
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="split",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
    ),
    split_size=_get_splits(allow_none=False, min_num_dims=1, allow_array_indices=False),
    dim=st.shared(
        helpers.get_axis(
            shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
            force_int=True,
        ),
        key="target_axis",
    ),
)
def test_torch_instance_split(
    dtype_value,
    split_size,
    dim,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "split_size": split_size,
            "dim": dim,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# tensor_split
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="tensor_split",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
    ),
    indices_or_sections=_get_splits(
        min_num_dims=1, allow_none=False, allow_array_indices=False
    ),
    dim=st.shared(
        helpers.get_axis(
            shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
            force_int=True,
        ),
        key="target_axis",
    ),
    method_num_positional_args=st.just(1),
)
def test_torch_instance_tensor_split(
    dtype_value,
    indices_or_sections,
    dim,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={
            "indices_or_sections": indices_or_sections,
            "dim": dim,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# vsplit
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="vsplit",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="value_shape"),
    ),
    indices_or_sections=_get_splits(
        min_num_dims=2,
        axis=0,
        allow_none=False,
        allow_array_indices=False,
        is_mod_split=True,
    ),
)
def test_torch_instance_vsplit(
    dtype_value,
    indices_or_sections,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={"indices_or_sections": indices_or_sections},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# hsplit
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="hsplit",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="value_shape"),
    ),
    indices_or_sections=_get_splits(
        min_num_dims=1,
        axis=1,
        allow_none=False,
        allow_array_indices=False,
        is_mod_split=True,
    ),
)
def test_torch_instance_hsplit(
    dtype_value,
    indices_or_sections,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={"indices_or_sections": indices_or_sections},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# dsplit
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="dsplit",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=3), key="value_shape"),
    ),
    indices_or_sections=_get_splits(
        min_num_dims=3,
        axis=2,
        allow_none=False,
        allow_array_indices=False,
        is_mod_split=True,
    ),
)
def test_torch_instance_dsplit(
    dtype_value,
    indices_or_sections,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={"indices_or_sections": indices_or_sections},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# detach
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="detach",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_torch_instance_detach(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# detach_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="detach_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_torch_instance_detach_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# dim
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="dim",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
)
def test_torch_instance_dim(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# ndimension
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="ndimension",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
)
def test_torch_instance_ndimension(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@st.composite
def _fill_value_and_size(
    draw,
    *,
    min_num_dims=1,
    max_num_dims=5,
    min_dim_size=1,
    max_dim_size=10,
):
    if isinstance(min_dim_size, st._internal.SearchStrategy):
        min_dim_size = draw(min_dim_size)
    if isinstance(max_dim_size, st._internal.SearchStrategy):
        max_dim_size = draw(max_dim_size)

    available_dtypes = draw(helpers.get_dtypes("numeric"))
    dtype = draw(
        helpers.array_dtypes(
            num_arrays=1,
            available_dtypes=available_dtypes,
        )
    )
    array = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=(1,),
        )
    )
    dtype.append("int32")
    size = draw(
        st.shared(
            helpers.get_shape(
                min_num_dims=min_num_dims,
                max_num_dims=max_num_dims,
                min_dim_size=min_dim_size,
                max_dim_size=max_dim_size,
            ),
            key="shape",
        )
    )
    fill_value = draw(helpers.ints()) if "int" in dtype[0] else draw(helpers.floats())

    return dtype, [array, size, fill_value]


# new_full
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="new_full",
    dtype_and_x=_fill_value_and_size(max_num_dims=3),
)
def test_torch_instance_new_full(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtype[0]],
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[input_dtype[1]],
        method_all_as_kwargs_np={
            "size": x[1],
            "fill_value": x[2],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# new_empty (not actually intuitive for testing)
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="new_empty",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
    size=helpers.get_shape(
        min_num_dims=1,
        max_num_dims=3,
    ),
)
def test_torch_instance_new_empty(
    dtype_and_x,
    size,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtype[0]],
        init_all_as_kwargs_np={
            "data": x,
        },
        method_input_dtypes=[ivy.int32],
        method_all_as_kwargs_np={
            "size": size,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@st.composite
def _expand_helper(draw):
    num_dims = draw(st.integers(min_value=1, max_value=10))
    shape = draw(
        helpers.get_shape(min_num_dims=num_dims, max_num_dims=num_dims).filter(
            lambda x: any(i == 1 for i in x)
        )
    )
    new_shape = draw(
        helpers.get_shape(min_num_dims=num_dims, max_num_dims=num_dims).filter(
            lambda x: all(x[i] == v if v != 1 else True for i, v in enumerate(shape))
        )
    )
    dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            shape=shape,
        )
    )
    return dtype, x, new_shape


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="expand",
    dtype_x_shape=_expand_helper(),
    unpack_shape=st.booleans(),
)
def test_torch_instance_expand(
    dtype_x_shape,
    unpack_shape,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x, shape = dtype_x_shape
    if unpack_shape:
        method_flags.num_positional_args = len(shape) + 1
        size = {}
        i = 0
        for x_ in shape:
            size["x{}".format(i)] = x_
            i += 1
    else:
        size = {
            "size": shape,
        }
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np=size,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# expand_as
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="expand_as",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"), num_arrays=2
    ),
)
def test_torch_instance_expand_as(
    dtype_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@st.composite
def _unfold_args(draw):
    values_dtype, values, axis, shape = draw(
        helpers.dtype_values_axis(
            available_dtypes=helpers.get_dtypes("float"),
            force_int_axis=True,
            shape=draw(
                helpers.get_shape(
                    allow_none=False,
                    min_num_dims=1,
                    min_dim_size=1,
                )
            ),
            ret_shape=True,
        )
    )
    size = draw(
        st.integers(
            min_value=1,
            max_value=max(shape[axis] - 1, 1),
        )
    )
    step = draw(
        st.integers(
            min_value=1,
            max_value=size,
        )
    )
    return values_dtype, values, axis, size, step


# unfold
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="unfold",
    dtype_values_args=_unfold_args(),
)
def test_torch_instance_unfold(
    dtype_values_args,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x, axis, size, step = dtype_values_args
    print(axis, size, step)
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x,
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dimension": axis,
            "size": size,
            "step": step,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __mod__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__mod__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
    ),
)
def test_torch_special_mod(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# long
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="long",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
    ),
)
def test_torch_instance_long(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# max
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="max",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_instance_max(
    dtype_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False)
    ).filter(lambda x: "bfloat16" not in x[0]),
)
def test_torch_tensor_property_is_quantized(
    dtype_x,
):
    _, data = dtype_x
    x = Tensor(data[0])
    x.ivy_array = data[0]
    ivy.utils.assertions.check_equal(
        x.is_quantized, "q" in ivy.dtype(ivy.array(data[0])), as_array=False
    )


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False)
    ).filter(lambda x: "bfloat16" not in x[0]),
)
def test_torch_tensor_property_is_cuda(
    dtype_x,
):
    _, data = dtype_x
    x = Tensor(data[0])
    x.ivy_array = data[0]
    ivy.utils.assertions.check_equal(
        x.is_cuda, "gpu" in ivy.dev(ivy.array(data[0])), as_array=False
    )


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False)
    ).filter(lambda x: "bfloat16" not in x[0]),
)
def test_torch_tensor_property_is_meta(
    dtype_x,
):
    _, data = dtype_x
    x = Tensor(data[0])
    x.ivy_array = data[0]
    ivy.utils.assertions.check_equal(
        x.is_meta, "meta" in ivy.dev(ivy.array(data[0])), as_array=False
    )


# logical_and
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="logical_and",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
    ),
)
def test_torch_instance_logical_and(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# logical_not
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="logical_not",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"), num_arrays=1
    ),
)
def test_torch_instance_logical_not(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# logical_or
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="logical_or",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
    ),
)
def test_torch_instance_logical_or(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# bitwise_not
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="bitwise_not",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
    ),
)
def test_torch_instance_bitwise_not(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        method_all_as_kwargs_np={},
        frontend=frontend,
        on_device=on_device,
    )


# bitwise_and
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="bitwise_and",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
    ),
)
def test_torch_instance_bitwise_and(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# bitwise_or
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="bitwise_or",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
    ),
)
def test_torch_instance_bitwise_or(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# bitwise_or_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="bitwise_or_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
    ),
)
def test_torch_instance_bitwise_or_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# bitwise_left_shift
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="bitwise_left_shift",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
    ),
)
def test_torch_instance_bitwise_left_shift(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# add_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="add_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
)
def test_torch_instance_add_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtype[0]],
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# subtract_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="subtract_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
)
def test_torch_instance_subtract_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtype[0]],
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# arccos_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="arccos_",
    dtype_and_x=helpers.dtype_and_values(
        min_value=-1.0,
        max_value=1.0,
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_instance_arccos_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# arccos
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="arccos",
    dtype_and_x=helpers.dtype_and_values(
        min_value=-1.0,
        max_value=1.0,
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_instance_arccos(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# acos_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="acos_",
    dtype_and_x=helpers.dtype_and_values(
        min_value=-1.0,
        max_value=1.0,
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_instance_acos_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# asin_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="asin_",
    dtype_and_x=helpers.dtype_and_values(
        min_value=-1.0,
        max_value=1.0,
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_instance_asin_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# arcsin_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="arcsin_",
    dtype_and_x=helpers.dtype_and_values(
        min_value=-1.0,
        max_value=1.0,
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_instance_arcsin_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# atan_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="atan_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_atan_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# tan_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="tan_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_tan_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# atanh
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="atanh",
    dtype_and_x=helpers.dtype_and_values(
        min_value=-1.0,
        max_value=1.0,
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_instance_atanh(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# atanh_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="atanh_",
    dtype_and_x=helpers.dtype_and_values(
        min_value=-1.0,
        max_value=1.0,
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_instance_atanh_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# arctanh
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="arctanh",
    dtype_and_x=helpers.dtype_and_values(
        min_value=-1.0,
        max_value=1.0,
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_instance_arctanh(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# arctanh_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="arctanh_",
    dtype_and_x=helpers.dtype_and_values(
        min_value=-1.0,
        max_value=1.0,
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_instance_arctanh_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# pow
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="pow",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch_instance_pow(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    dtype = input_dtype[0]
    if "int" in dtype:
        x[1] = ivy.abs(x[1])
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "exponent": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# pow_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="pow_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
)
def test_torch_instance_pow_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    dtype = input_dtype[0]
    if "int" in dtype:
        x[1] = ivy.abs(x[1])
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "exponent": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __pow__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__pow__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
)
def test_torch_special_pow(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    dtype = input_dtype[0]
    if "int" in dtype:
        x[1] = ivy.abs(x[1])
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "exponent": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __rpow__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__rpow__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        min_value=1,
    ),
)
def test_torch_special_rpow(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    dtype = input_dtype[0]
    if "int" in dtype:
        x[0] = ivy.abs(x[0])
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# arccosh_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="arccosh_",
    dtype_and_x=helpers.dtype_and_values(
        min_value=-1.0,
        max_value=1.0,
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_instance_arccosh_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# argmax
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="argmax",
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        force_int_axis=True,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
        min_value=1,
        max_value=5,
        valid_axis=True,
        allow_neg_axes=True,
    ),
    keepdim=st.booleans(),
)
def test_torch_instance_argmax(
    dtype_input_axis,
    keepdim,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x, axis = dtype_input_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dim": axis,
            "keepdim": keepdim,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# argmin
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="argmin",
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        force_int_axis=True,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
        min_value=1,
        max_value=5,
        valid_axis=True,
        allow_neg_axes=True,
    ),
    keepdim=st.booleans(),
)
def test_torch_instance_argmin(
    dtype_input_axis,
    keepdim,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x, axis = dtype_input_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dim": axis,
            "keepdim": keepdim,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# argsort
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="argsort",
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        force_int_axis=True,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
        min_value=1,
        max_value=5,
        valid_axis=True,
        allow_neg_axes=True,
    ),
    descending=st.booleans(),
)
def test_torch_instance_argsort(
    dtype_input_axis,
    descending,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x, axis = dtype_input_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dim": axis,
            "descending": descending,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# arccosh
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="arccosh",
    dtype_and_x=helpers.dtype_and_values(
        min_value=-1.0,
        max_value=1.0,
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_instance_arccosh(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# ceil
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="ceil",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_instance_ceil(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# argwhere
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="argwhere",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_torch_instance_argwhere(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# size
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="size",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ),
    dim=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
        force_int=True,
    ),
)
def test_torch_instance_size(
    dtype_and_x,
    dim,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dim": dim,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# min
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="min",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_instance_min(
    dtype_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@st.composite
def _get_dtype_and_multiplicative_matrices(draw):
    return draw(
        st.one_of(
            _get_dtype_input_and_matrices(),
            _get_dtype_and_3dbatch_matrices(),
        )
    )


# matmul
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="matmul",
    dtype_tensor1_tensor2=_get_dtype_and_multiplicative_matrices(),
)
def test_torch_instance_matmul(
    dtype_tensor1_tensor2,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    dtype, tensor1, tensor2 = dtype_tensor1_tensor2
    helpers.test_frontend_method(
        init_input_dtypes=dtype,
        init_all_as_kwargs_np={
            "data": tensor1,
        },
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={"other": tensor2},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@st.composite
def _array_idxes_n_dtype(draw, **kwargs):
    num_dims = draw(helpers.ints(min_value=1, max_value=4))
    dtype, x = draw(
        helpers.dtype_and_values(
            **kwargs, min_num_dims=num_dims, max_num_dims=num_dims, shared_dtype=True
        )
    )
    idxes = draw(
        st.lists(
            helpers.ints(min_value=0, max_value=num_dims - 1),
            min_size=num_dims,
            max_size=num_dims,
            unique=True,
        )
    )
    return x, idxes, dtype


# permute
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="permute",
    dtype_values_axis=_array_idxes_n_dtype(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_instance_permute(
    dtype_values_axis,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    x, idxes, dtype = dtype_values_axis
    unpack_dims = True
    if unpack_dims:
        method_flags.num_positional_args = len(idxes) + 1
        dims = {}
        i = 0
        for x_ in idxes:
            dims["x{}".format(i)] = x_
            i += 1
    else:
        dims = {
            "dims": tuple(idxes),
        }
    helpers.test_frontend_method(
        init_input_dtypes=dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=dtype,
        method_all_as_kwargs_np=dims,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# mean
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="mean",
    dtype_and_x=_statistical_dtype_values(
        function="mean",
        min_value=-1e04,
        max_value=1e04,
    ),
    keepdims=st.booleans(),
)
def test_torch_instance_mean(
    dtype_and_x,
    keepdims,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x, axis = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dim": axis,
            "keepdim": keepdims,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# nanmean
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="nanmean",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-1e04,
        max_value=1e04,
    ),
)
def test_torch_instance_nanmean(
    dtype_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# median
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="median",
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
    ),
    keepdim=st.booleans(),
)
def test_torch_instance_median(
    dtype_input_axis,
    keepdim,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x, axis = dtype_input_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dim": axis,
            "keepdim": keepdim,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# transpose
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="transpose",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ),
    dim0=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="shape"),
        allow_neg=True,
        force_int=True,
    ),
    dim1=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="shape"),
        allow_neg=True,
        force_int=True,
    ),
)
def test_torch_instance_transpose(
    dtype_value,
    dim0,
    dim1,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"dim0": dim0, "dim1": dim1},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# transpose_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="transpose_",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ),
    dim0=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="shape"),
        allow_neg=True,
        force_int=True,
    ),
    dim1=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="shape"),
        allow_neg=True,
        force_int=True,
    ),
)
def test_torch_instance_transpose_(
    dtype_value,
    dim0,
    dim1,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dim0": dim0,
            "dim1": dim1,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# t
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="t",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=helpers.get_shape(min_num_dims=2, max_num_dims=2),
    ),
)
def test_torch_instance_t(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# flatten
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="flatten",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(), key="shape"),
    ),
    axes=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="shape"),
        min_size=2,
        max_size=2,
        unique=False,
        force_tuple=True,
    ),
)
def test_torch_instance_flatten(
    dtype_value,
    axes,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "start_dim": axes[0],
            "end_dim": axes[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# cumsum
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="cumsum",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ),
    dim=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="shape"),
        allow_neg=True,
        force_int=True,
    ),
    dtypes=_dtypes(),
)
def test_torch_instance_cumsum(
    dtype_value,
    dim,
    dtypes,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=dtypes,
        method_all_as_kwargs_np={
            "dim": dim,
            "dtype": dtypes[0],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# cumsum_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="cumsum_",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ),
    dim=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="shape"),
        allow_neg=True,
        force_int=True,
    ),
)
def test_torch_instance_cumsum_(
    dtype_value,
    dim,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dim": dim,
            "dtype": input_dtype[0],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# sort
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="sort",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ),
    dim=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="shape"),
        allow_neg=True,
        force_int=True,
    ),
    descending=st.booleans(),
)
def test_torch_instance_sort(
    dtype_value,
    dim,
    descending,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dim": dim,
            "descending": descending,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# sigmoid
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="sigmoid",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_instance_sigmoid(
    dtype_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# sigmoid
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="sigmoid_",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_instance_sigmoid_(
    dtype_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# softmax
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="softmax",
    dtype_x_and_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        max_axes_size=1,
        force_int_axis=True,
        valid_axis=True,
    ),
    dtype=helpers.get_dtypes("float", full=False),
)
def test_torch_instance_softmax(
    dtype_x_and_axis,
    dtype,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x, axis = dtype_x_and_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dim": axis,
            "dtype": dtype[0],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@st.composite
def _repeat_helper(draw):
    shape = draw(
        helpers.get_shape(
            min_num_dims=1, max_num_dims=5, min_dim_size=2, max_dim_size=10
        )
    )

    input_dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            shape=shape,
        )
    )

    repeats = draw(st.lists(st.integers(min_value=1, max_value=5), min_size=len(shape)))
    return input_dtype, x, repeats


# repeat
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="repeat",
    dtype_x_repeats=_repeat_helper(),
    unpack_repeat=st.booleans(),
)
def test_torch_instance_repeat(
    dtype_x_repeats,
    unpack_repeat,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x, repeats = dtype_x_repeats
    repeat = {
        "repeats": repeats,
    }
    if unpack_repeat:
        method_flags.num_positional_args = len(repeat["repeats"]) + 1
        for i, x_ in enumerate(repeat["repeats"]):
            repeat["x{}".format(i)] = x_
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np=repeat,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# unbind
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="unbind",
    dtype_value_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
    ),
)
def test_torch_instance_unbind(
    dtype_value_axis,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtypes, x, axis = dtype_value_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "dim": axis,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __eq__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__eq__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch_special_eq(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# inverse
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="inverse",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=2,
    ).filter(lambda s: s[1][0].shape[-1] == s[1][0].shape[-2]),
)
def test_torch_instance_inverse(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# neg
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="neg",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch_instance_neg(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __neg__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__neg__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch_special_neg(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# int
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="int",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
    ),
)
def test_torch_instance_int(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# half
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="half",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_torch_instance_half(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# bool
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="bool",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
    ),
)
def test_torch_instance_bool(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# type
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="type",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    dtype=helpers.get_dtypes("valid", full=False),
)
def test_torch_instance_type(
    dtype_and_x,
    dtype,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dtype": dtype[0],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# type_as
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="type_as",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
    ),
)
def test_torch_instance_type_as(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# byte
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="byte",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_torch_instance_byte(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# ne
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="ne",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch_instance_ne(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# squeeze
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="squeeze",
    dtype_value_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ),
)
def test_torch_instance_squeeze(
    dtype_value_axis,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x, axis = dtype_value_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dim": axis,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# squeeze_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="squeeze_",
    dtype_value_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ),
)
def test_torch_instance_squeeze_(
    dtype_value_axis,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x, axis = dtype_value_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dim": axis,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# flip
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="flip",
    dtype_values_axis=_array_idxes_n_dtype(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_instance_flip(
    dtype_values_axis,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    x, idxes, dtype = dtype_values_axis
    helpers.test_frontend_method(
        init_input_dtypes=dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={
            "dims": idxes,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# fliplr
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="fliplr",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=2,
    ),
)
def test_torch_instance_fliplr(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# tril
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="tril",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,  # Torch requires this.
    ),
    diagonal=st.integers(min_value=-100, max_value=100),
)
def test_torch_instance_tril(
    dtype_and_values,
    diagonal,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_values
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "diagonal": diagonal,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# tril_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="tril_",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,  # Torch requires this.
    ),
    diagonal=st.integers(min_value=-100, max_value=100),
)
def test_torch_instance_tril_(
    dtype_and_values,
    diagonal,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_values
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "diagonal": diagonal,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# sqrt
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="sqrt",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
)
def test_torch_instance_sqrt(
    dtype_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# sqrt_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="sqrt_",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_instance_sqrt_(
    dtype_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# index_select
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="index_select",
    params_indices_others=helpers.array_indices_axis(
        array_dtypes=helpers.get_dtypes("valid"),
        indices_dtypes=["int64"],
        max_num_dims=1,
        indices_same_dims=True,
    ),
)
def test_torch_instance_index_select(
    params_indices_others,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtypes, input, indices, axis, batch_dims = params_indices_others
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtypes[0]],
        init_all_as_kwargs_np={
            "data": input,
        },
        method_input_dtypes=[input_dtypes[1]],
        method_all_as_kwargs_np={
            "dim": axis,
            "index": indices,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@st.composite
def _arrays_dim_idx_n_dtypes(draw):
    num_dims = draw(st.shared(helpers.ints(min_value=1, max_value=4), key="num_dims"))
    num_arrays = 2
    common_shape = draw(
        helpers.lists(
            x=helpers.ints(min_value=2, max_value=3),
            min_size=num_dims - 1,
            max_size=num_dims - 1,
        )
    )
    _dim = draw(helpers.ints(min_value=0, max_value=num_dims - 1))
    unique_dims = draw(
        helpers.lists(
            x=helpers.ints(min_value=2, max_value=3),
            min_size=num_arrays,
            max_size=num_arrays,
        )
    )

    min_dim = min(unique_dims)
    max_dim = max(unique_dims)
    _idx = draw(
        helpers.array_values(
            shape=min_dim,
            dtype="int64",
            min_value=0,
            max_value=max_dim,
            exclude_min=False,
        )
    )

    xs = list()
    available_input_types = draw(helpers.get_dtypes("numeric"))
    input_dtypes = draw(
        helpers.array_dtypes(
            available_dtypes=available_input_types,
            num_arrays=num_arrays,
            shared_dtype=True,
        )
    )
    for ud, dt in zip(unique_dims, input_dtypes):
        x = draw(
            helpers.array_values(
                shape=common_shape[:_dim] + [ud] + common_shape[_dim:],
                dtype=dt,
                large_abs_safety_factor=2.5,
                small_abs_safety_factor=2.5,
                safety_factor_scale="log",
            )
        )
        xs.append(x)
    return xs, input_dtypes, _dim, _idx


# index_add
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="index_add_",
    xs_dtypes_dim_idx=_arrays_dim_idx_n_dtypes(),
    alpha=st.integers(min_value=1, max_value=2),
)
def test_torch_instance_index_add_(
    *,
    xs_dtypes_dim_idx,
    alpha,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    frontend,
):
    xs, input_dtypes, axis, indices = xs_dtypes_dim_idx
    if xs[0].shape[axis] < xs[1].shape[axis]:
        source, input = xs
    else:
        input, source = xs
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtypes[0]],
        init_all_as_kwargs_np={
            "data": input,
        },
        method_input_dtypes=["int64", input_dtypes[1]],
        method_all_as_kwargs_np={
            "dim": axis,
            "index": indices,
            "source": source,
            "alpha": alpha,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
        rtol_=1e-03,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="index_add",
    xs_dtypes_dim_idx=_arrays_dim_idx_n_dtypes(),
    alpha=st.integers(min_value=1, max_value=2),
)
def test_torch_instance_index_add(
    *,
    xs_dtypes_dim_idx,
    alpha,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    frontend,
):
    xs, input_dtypes, axis, indices = xs_dtypes_dim_idx
    if xs[0].shape[axis] < xs[1].shape[axis]:
        source, input = xs
    else:
        input, source = xs
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtypes[0]],
        init_all_as_kwargs_np={
            "data": input,
        },
        method_input_dtypes=["int64", input_dtypes[1]],
        method_all_as_kwargs_np={
            "dim": axis,
            "index": indices,
            "source": source,
            "alpha": alpha,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
        rtol_=1e-03,
    )


@st.composite
def _get_clamp_inputs(draw):
    shape = draw(
        helpers.get_shape(
            min_num_dims=1, max_num_dims=5, min_dim_size=2, max_dim_size=10
        )
    )
    x_dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"),
            shape=shape,
        )
    )
    min = draw(st.booleans())
    if min:
        max = draw(st.booleans())
        min = draw(
            helpers.array_values(
                dtype=x_dtype[0], shape=shape, min_value=0, max_value=25
            )
        )
        max = (
            draw(
                helpers.array_values(
                    dtype=x_dtype[0], shape=shape, min_value=26, max_value=50
                )
            )
            if max
            else None
        )
    else:
        min = None
        max = draw(
            helpers.array_values(
                dtype=x_dtype[0], shape=shape, min_value=26, max_value=50
            )
        )
    return x_dtype, x, min, max


# clamp
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="clamp",
    dtype_and_x_min_max=_get_clamp_inputs(),
)
def test_torch_instance_clamp(
    dtype_and_x_min_max,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x, min, max = dtype_and_x_min_max
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"min": min, "max": max},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# clamp_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="clamp_",
    dtype_and_x_min_max=_get_clamp_inputs(),
)
def test_torch_instance_clamp_(
    dtype_and_x_min_max,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x, min, max = dtype_and_x_min_max
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"min": min, "max": max},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# clip
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="clip",
    input_and_ranges=_get_clamp_inputs(),
)
def test_torch_instance_clip(
    input_and_ranges,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x, min, max = input_and_ranges
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"min": min, "max": max},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# clip_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="clip_",
    input_and_ranges=_get_clamp_inputs(),
)
def test_torch_instance_clip_(
    input_and_ranges,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x, min, max = input_and_ranges
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"min": min, "max": max},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __gt__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__gt__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch_special_gt(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __ne__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__ne__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch_special_ne(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __lt__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__lt__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch_special_lt(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __or__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__or__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch_special_or(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# where
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="where",
    broadcastables=_broadcastable_trio(),
)
def test_torch_instance_where(
    broadcastables,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    cond, xs, dtypes = broadcastables
    helpers.test_frontend_method(
        init_input_dtypes=dtypes,
        init_all_as_kwargs_np={
            "data": xs[0],
        },
        method_input_dtypes=["bool", dtypes[1]],
        method_all_as_kwargs_np={
            "condition": cond,
            "other": xs[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# clone
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="clone",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=1,
    ),
)
def test_torch_instance_clone(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __invert__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__invert__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=1,
    ),
)
def test_torch_special_invert(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# acosh
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="acosh",
    dtype_and_x=helpers.dtype_and_values(
        min_value=1.0,
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_instance_acosh(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@st.composite
def _masked_fill_helper(draw):
    cond, xs, dtypes = draw(_broadcastable_trio())
    if ivy.is_uint_dtype(dtypes[0]):
        fill_value = draw(helpers.ints(min_value=0, max_value=5))
    elif ivy.is_int_dtype(dtypes[0]):
        fill_value = draw(helpers.ints(min_value=-5, max_value=5))
    else:
        fill_value = draw(helpers.floats(min_value=-5, max_value=5))
    return dtypes[0], xs[0], cond, fill_value


# masked_fill
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="masked_fill",
    x_mask_val=_masked_fill_helper(),
)
def test_torch_instance_masked_fill(
    x_mask_val,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    dtype, x, mask, val = x_mask_val
    helpers.test_frontend_method(
        init_input_dtypes=[dtype],
        init_all_as_kwargs_np={
            "data": x,
        },
        method_input_dtypes=["bool", dtype],
        method_all_as_kwargs_np={
            "mask": mask,
            "value": val,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# acosh_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="acosh_",
    dtype_and_x=helpers.dtype_and_values(
        min_value=1.0,
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_instance_acosh_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# numpy
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="numpy",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_torch_instance_numpy(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    ret, frontend_ret = helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
        test_values=False,
    )
    # manual testing required as function return is numpy frontend
    helpers.value_test(
        ret_np_flat=helpers.flatten_and_to_np(ret=ret),
        ret_np_from_gt_flat=frontend_ret[0],
        ground_truth_backend="torch",
    )


# atan2_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="atan2_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
    ),
)
def test_torch_instance_atan2_(
    dtype_and_x, frontend_method_data, init_flags, method_flags, frontend, on_device
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# bitwise_not_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="bitwise_not_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
    ),
)
def test_torch_instance_bitwise_not_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        method_all_as_kwargs_np={},
        frontend=frontend,
        on_device=on_device,
    )


# bitwise_and_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="bitwise_and_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
    ),
)
def test_torch_instance_bitwise_and_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __and__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__and__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch_special_and(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# bitwise_xor
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="bitwise_xor",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
        num_arrays=2,
    ),
)
def test_torch_instance_bitwise_xor(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# cumprod
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="cumprod",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ),
    dim=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="shape"),
        allow_neg=True,
        force_int=True,
    ),
    dtypes=_dtypes(),
)
def test_torch_instance_cumprod(
    dtype_value,
    dim,
    dtypes,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=dtypes,
        method_all_as_kwargs_np={
            "dim": dim,
            "dtype": dtypes[0],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# relu
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="relu",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_relu(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# fmin
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="fmin",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
    ),
)
def test_torch_instance_fmin(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# count_nonzero
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="count_nonzero",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ),
    dim=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="shape"),
        allow_neg=True,
        force_int=True,
    ),
)
def test_torch_instance_count_nonzero(
    dtype_value,
    dim,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"dim": dim},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# exp
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="exp",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
)
def test_torch_instance_exp(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# exp_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="exp_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
)
def test_torch_instance_exp_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# expm1
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="expm1",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
)
def test_torch_instance_expm1(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# mul
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="mul",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
)
def test_torch_instance_mul(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# ceil_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="ceil_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_instance_ceil_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# mul_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="mul_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_torch_instance_mul_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# trunc
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="trunc",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ),
)
def test_torch_instance_trunc(
    dtype_value,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# trunc_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="trunc_",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ),
)
def test_torch_instance_trunc_(
    dtype_value,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# fix
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="fix",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ),
)
def test_torch_instance_fix(
    dtype_value,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# fix_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="fix_",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ),
)
def test_torch_instance_fix_(
    dtype_value,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# round
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="round",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    decimals=st.integers(min_value=0, max_value=5),
)
def test_torch_instance_round(
    dtype_and_x,
    decimals,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "decimals": decimals,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# cross
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="cross",
    dtype_input_other_dim=dtype_value1_value2_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=10,
        min_dim_size=3,
        max_dim_size=3,
        min_value=-1e10,
        max_value=1e10,
        abs_smallest_val=0.01,
        large_abs_safety_factor=2,
        safety_factor_scale="log",
    ),
)
def test_torch_instance_cross(
    dtype_input_other_dim,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    dtype, input, other, dim = dtype_input_other_dim
    helpers.test_frontend_method(
        init_input_dtypes=dtype,
        init_all_as_kwargs_np={
            "data": input,
        },
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={
            "other": other,
            "dim": dim,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
        rtol_=1e-2,
        atol_=1e-2,
    )


# det
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="det",
    dtype_and_x=_get_dtype_and_matrix(square=True, batch=True),
)
def test_torch_instance_det(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x,
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# reciprocal
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="reciprocal",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=1,
    ),
)
def test_torch_instance_reciprocal(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# fill_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="fill_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    value=helpers.floats(min_value=1, max_value=10),
)
def test_torch_instance_fill_(
    dtype_and_x,
    value,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "value": value,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# nonzero
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="nonzero",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
)
def test_torch_instance_nonzero(
    dtype_and_values,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_values
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# mm
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="mm",
    dtype_xy=_get_dtype_input_and_matrices(),
)
def test_torch_instance_mm(
    dtype_xy,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    dtype, x, y = dtype_xy
    helpers.test_frontend_method(
        init_input_dtypes=dtype,
        init_all_as_kwargs_np={
            "data": x,
        },
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={
            "mat2": y,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# square
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="square",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_instance_square(
    dtype_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# log10
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="log10",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_log10(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# log10_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="log10_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_log10_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# short
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="short",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch_instance_short(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# prod
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="prod",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=5,
        valid_axis=True,
        allow_neg_axes=False,
        max_axes_size=1,
        force_int_axis=True,
        large_abs_safety_factor=10,
        small_abs_safety_factor=10,
        safety_factor_scale="log",
    ),
    dtype=helpers.get_dtypes("float", none=True, full=False),
    keepdims=st.booleans(),
)
def test_torch_instance_prod(
    dtype_x_axis,
    dtype,
    keepdims,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x, axis = dtype_x_axis
    if ivy.current_backend_str() == "torch":
        init_flags.as_variable = [False]
        method_flags.as_variable = [False]
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dim": axis,
            "keepdim": keepdims,
            "dtype": dtype[0],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# div
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="div",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
    ),
    rounding_mode=st.sampled_from(["floor", "trunc"]) | st.none(),
)
def test_torch_instance_div(
    dtype_and_x,
    rounding_mode,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x[1], 0)))

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
            "rounding_mode": rounding_mode,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# div_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="div_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
    ),
    rounding_mode=st.sampled_from(["floor", "trunc"]) | st.none(),
)
def test_torch_instance_div_(
    dtype_and_x,
    rounding_mode,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x[1], 0)))

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
            "rounding_mode": rounding_mode,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# normal_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="normal_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    mean=helpers.floats(min_value=-1, max_value=1),
    std=helpers.floats(min_value=0, max_value=1),
)
def test_torch_instance_normal_(
    dtype_and_x,
    mean,
    std,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    dtype, x = dtype_and_x

    def call():
        return helpers.test_frontend_method(
            init_input_dtypes=dtype,
            init_all_as_kwargs_np={"data": x[0]},
            method_input_dtypes=dtype,
            method_all_as_kwargs_np={
                "mean": mean,
                "std": std,
            },
            frontend_method_data=frontend_method_data,
            init_flags=init_flags,
            method_flags=method_flags,
            frontend=frontend,
            on_device=on_device,
            test_values=False,
        )

    ret = call()

    if not ivy.exists(ret):
        return

    ret_np, ret_from_np = ret
    ret_np = helpers.flatten_and_to_np(ret=ret_np)
    ret_from_np = helpers.flatten_and_to_np(ret=ret_from_np)
    for u, v in zip(ret_np, ret_from_np):
        assert u.dtype == v.dtype
        assert u.shape == v.shape


# addcdiv
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="addcdiv",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=3,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
    value=st.floats(min_value=-100, max_value=100),
)
def test_torch_instance_addcdiv(
    dtype_and_x,
    value,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x[2], 0)))

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "tensor1": x[1],
            "tensor2": x[2],
            "value": value,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
        atol_=1e-03,
    )


# addcmul
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="addcmul",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=3,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
    value=st.floats(min_value=-100, max_value=100),
)
def test_torch_instance_addcmul(
    dtype_and_x,
    value,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "tensor1": x[1],
            "tensor2": x[2],
            "value": value,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
        atol_=1e-02,
    )


# addcmul_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="addcmul_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=3,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
    value=st.floats(min_value=-100, max_value=100),
)
def test_torch_instance_addcmul_(
    dtype_and_x,
    value,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "tensor1": x[1],
            "tensor2": x[2],
            "value": value,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
        atol_=1e-02,
    )


# sign
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="sign",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_torch_instance_sign(
    dtype_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# sign_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="sign_",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_torch_instance_sign_(
    dtype_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=[input_dtype],
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
        frontend=frontend,
    )


# std
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="std",
    dtype_and_x=_statistical_dtype_values(function="std"),
)
def test_torch_instance_std(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x, _, _ = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# fmod
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="fmod",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
        min_num_dims=1,
        min_value=-100,
        max_value=100,
    ),
)
def test_torch_instance_fmod(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"other": x[1]},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# fmod_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="fmod_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
        min_num_dims=1,
        min_value=-100,
        max_value=100,
    ),
)
def test_torch_instance_fmod_(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"other": x[1]},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# topk
# TODO: add value test after the stable sorting is added to torch
# https://github.com/pytorch/pytorch/issues/88184
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="topk",
    dtype_x_axis_k=_topk_helper(),
    largest=st.booleans(),
    sorted=st.booleans(),
)
def test_torch_instance_topk(
    dtype_x_axis_k,
    largest,
    sorted,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, input, axis, k = dtype_x_axis_k
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"data": input[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "k": k,
            "dim": axis,
            "largest": largest,
            "sorted": sorted,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
        test_values=False,
    )


# bitwise right shift
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="bitwise_right_shift",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_torch_instance_bitwise_right_shift(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    # negative shifts will throw an exception
    # shifts >= dtype witdth produce backend-defined behavior
    x[1] = np.asarray(
        np.clip(x[1], 0, np.iinfo(input_dtype[1]).bits - 1), dtype=input_dtype[1]
    )
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# logdet
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="logdet",
    dtype_and_x=_get_dtype_and_matrix(square=True, batch=True),
)
def test_torch_instance_logdet(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    dtype, x = dtype_and_x
    x = np.matmul(x.T, x) + np.identity(x.shape[0])
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x,
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# multiply
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="multiply",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
)
def test_torch_instance_multiply(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# norm
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="norm",
    p_dtype_x_axis=_get_axis_and_p(),
    keepdim=st.booleans(),
    dtype=helpers.get_dtypes("valid", full=False),
)
def test_torch_instance_norm(
    p_dtype_x_axis,
    keepdim,
    dtype,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    p, values = p_dtype_x_axis
    input_dtype, x, axis = values
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "p": p,
            "dim": axis,
            "keepdim": keepdim,
            "dtype": dtype[0],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# isinf
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="isinf",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_torch_instance_isinf(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# is_complex
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="is_complex",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_torch_instance_is_complex(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# copysign
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="copysign",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        num_arrays=2,
    ),
)
def test_torch_instance_copysign(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# not_equal
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="not_equal",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
    ),
)
def test_torch_instance_not_equal(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        atol_=1e-02,
        on_device=on_device,
    )


@st.composite
def _get_dtype_input_and_vectors(draw, with_input=False, same_size=False):
    dim_size1 = draw(helpers.ints(min_value=2, max_value=5))
    dim_size2 = dim_size1 if same_size else draw(helpers.ints(min_value=2, max_value=5))
    dtype = draw(helpers.get_dtypes("float", full=True))
    dtype = [
        draw(st.sampled_from(tuple(set(dtype).difference({"bfloat16", "float16"}))))
    ]
    vec1 = draw(
        helpers.array_values(
            dtype=dtype[0], shape=(dim_size1,), min_value=2, max_value=5
        )
    )
    vec2 = draw(
        helpers.array_values(
            dtype=dtype[0], shape=(dim_size2,), min_value=2, max_value=5
        )
    )
    if with_input:
        input = draw(
            helpers.array_values(
                dtype=dtype[0], shape=(dim_size1, dim_size2), min_value=2, max_value=5
            )
        )
        return dtype, input, vec1, vec2
    return dtype, vec1, vec2


# addr
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="addr",
    dtype_and_vecs=_get_dtype_input_and_vectors(with_input=True),
    beta=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
    alpha=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
)
def test_torch_instance_addr(
    dtype_and_vecs,
    beta,
    alpha,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    dtype, input, vec1, vec2 = dtype_and_vecs
    helpers.test_frontend_method(
        init_input_dtypes=dtype,
        init_all_as_kwargs_np={
            "data": input,
        },
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={
            "vec1": vec1,
            "vec2": vec2,
            "beta": beta,
            "alpha": alpha,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        atol_=1e-02,
        on_device=on_device,
    )


# logical_not_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="logical_not_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=1,
        large_abs_safety_factor=12,
    ),
)
def test_torch_instance_logical_not_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# rsqrt
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="rsqrt",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_instance_rsqrt(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# equal
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="equal",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
        min_num_dims=1,
        min_value=-1e04,
        max_value=1e04,
    ),
)
def test_torch_instance_equal(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        atol_=1e-04,
        rtol_=1e-04,
        on_device=on_device,
    )


# erf
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="erf",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_instance_erf(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# greater
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="greater",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch_instance_greater(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# greater_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="greater_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch_instance_greater_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# greater_equal
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="greater_equal",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch_instance_greater_equal(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# greater_equal_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="greater_equal_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch_instance_greater_equal_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# less
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="less",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch_instance_less(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# less_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="less_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch_instance_less_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# less_equal
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="less_equal",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch_instance_less_equal(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# less_equal_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="less_equal_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch_instance_less_equal_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# addr_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="addr_",
    dtype_and_vecs=_get_dtype_input_and_vectors(with_input=True),
    beta=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
    alpha=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
)
def test_torch_instance_addr_(
    dtype_and_vecs,
    beta,
    alpha,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    dtype, input, vec1, vec2 = dtype_and_vecs
    helpers.test_frontend_method(
        init_input_dtypes=dtype,
        init_all_as_kwargs_np={
            "data": input,
        },
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={
            "vec1": vec1,
            "vec2": vec2,
            "beta": beta,
            "alpha": alpha,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        atol_=1e-02,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="eq_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch_special_eq_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="var",
    dtype_and_x=_statistical_dtype_values(
        function="var",
        min_value=-1e04,
        max_value=1e04,
    ),
    keepdim=st.booleans(),
)
def test_torch_instance_var(
    dtype_and_x,
    keepdim,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x, axis, correction = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dim": axis,
            "correction": int(correction),
            "keepdim": keepdim,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="narrow",
    dtype_input_dim_start_length=_dtype_input_dim_start_length(),
)
def test_torch_instance_narrow(
    dtype_input_dim_start_length,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    (input_dtype, x, dim, start, length) = dtype_input_dim_start_length
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dim": dim,
            "start": start,
            "length": length,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="as_strided",
    dtype_x_and_other=_as_strided_helper(),
)
def test_torch_instance_as_strided(
    dtype_x_and_other,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x, size, stride, offset = dtype_x_and_other
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "size": size,
            "stride": stride,
            "storage_offset": offset,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="stride",
    dtype_value_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
    ),
)
def test_torch_instance_stride(
    dtype_value_axis,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x, axis = dtype_value_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"dim": axis},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="log1p",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_log1p(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="log1p_",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
)
def test_torch_instance_log1p_(
    dtype_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="baddbmm",
    dtype_and_matrices=_get_dtype_and_3dbatch_matrices(with_input=True, input_3d=True),
    beta=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
    alpha=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
)
def test_torch_instance_baddbmm(
    dtype_and_matrices,
    beta,
    alpha,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x, batch1, batch2 = dtype_and_matrices
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "batch1": batch1,
            "batch2": batch2,
            "beta": beta,
            "alpha": alpha,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="floor_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_instance_floor_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="diag",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=st.shared(helpers.get_shape(min_num_dims=1, max_num_dims=2), key="shape"),
    ),
    diagonal=st.integers(min_value=-100, max_value=100),
)
def test_torch_instance_diag(
    dtype_and_values,
    diagonal,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, values = dtype_and_values
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": values[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "diagonal": diagonal,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="gather",
    params_indices_others=helpers.array_indices_axis(
        array_dtypes=helpers.get_dtypes("valid"),
        indices_dtypes=["int64"],
        indices_same_dims=True,
    ),
)
def test_torch_instance_gather(
    params_indices_others,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtypes, x, indices, axis, batch_dims = params_indices_others
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtypes[0]],
        init_all_as_kwargs_np={"data": x},
        method_input_dtypes=[input_dtypes[1]],
        method_all_as_kwargs_np={
            "dim": axis,
            "index": indices,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="take_along_dim",
    dtype_indices_axis=helpers.array_indices_axis(
        array_dtypes=helpers.get_dtypes("numeric"),
        indices_dtypes=["int64"],
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
        indices_same_dims=True,
    ),
)
def test_torch_instance_take_along_dim(
    dtype_indices_axis,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtypes, value, indices, axis, _ = dtype_indices_axis
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtypes[0]],
        init_all_as_kwargs_np={
            "data": value,
        },
        method_input_dtypes=[input_dtypes[1]],
        method_all_as_kwargs_np={
            "indices": indices,
            "dim": axis,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="movedim",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-100,
        max_value=100,
        shape=st.shared(
            helpers.get_shape(
                min_num_dims=1,
                max_num_dims=3,
                min_dim_size=1,
                max_dim_size=3,
            ),
            key="a_s_d",
        ),
    ),
    source=helpers.get_axis(
        allow_none=False,
        unique=True,
        shape=st.shared(
            helpers.get_shape(
                min_num_dims=1,
                max_num_dims=3,
                min_dim_size=1,
                max_dim_size=3,
            ),
            key="a_s_d",
        ),
        min_size=1,
        force_int=True,
    ),
    destination=helpers.get_axis(
        allow_none=False,
        unique=True,
        shape=st.shared(
            helpers.get_shape(
                min_num_dims=1,
                max_num_dims=3,
                min_dim_size=1,
                max_dim_size=3,
            ),
            key="a_s_d",
        ),
        min_size=1,
        force_int=True,
    ),
)
def test_torch_instance_movedim(
    dtype_and_input,
    source,
    destination,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, value = dtype_and_input
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"data": value[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "source": source,
            "destination": destination,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="addcdiv_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=3,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
    value=st.floats(min_value=-100, max_value=100),
)
def test_torch_instance_addcdiv_(
    dtype_and_x,
    value,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x[2], 0)))

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "tensor1": x[1],
            "tensor2": x[2],
            "value": value,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
        atol_=1e-03,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="cholesky",
    dtype_and_x=_get_dtype_and_matrix(square=True),
    upper=st.booleans(),
)
def test_torch_instance_cholesky(
    dtype_and_x,
    upper,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_and_x
    x = x[0]
    # make symmetric positive-definite
    x = np.matmul(x.swapaxes(-1, -2), x) + np.identity(x.shape[-1]) * 1e-3

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x,
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "upper": upper,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
        rtol_=1e-2,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="heaviside",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
    ),
)
def test_torch_instance_heaviside(
    dtype_and_values,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, values = dtype_and_values
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": values[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "values": values[1],
        },
        init_flags=init_flags,
        method_flags=method_flags,
        frontend_method_data=frontend_method_data,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="dot",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shape=(1,),
    ),
)
def test_torch_instance_dot(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "tensor": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="tile",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(), key="shape"),
    ),
    reps=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="shape"),
        allow_neg=False,
    ),
)
def test_torch_instance_tile(
    dtype_and_values,
    reps,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, values = dtype_and_values
    if isinstance(reps, tuple):
        method_flags.num_positional_args = len(reps)
    else:
        method_flags.num_positional_args = 1
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": values[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "reps": reps,
        },
        init_flags=init_flags,
        method_flags=method_flags,
        frontend_method_data=frontend_method_data,
        frontend=frontend,
        on_device=on_device,
    )


# write test for torch instance apply_


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="apply_",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=1,
    ),
)
def test_torch_instance_apply_(
    dtype_and_values,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    def func(x):
        return x + 1

    input_dtype, values = dtype_and_values

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": values[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "callable": func,
        },
        init_flags=init_flags,
        method_flags=method_flags,
        frontend_method_data=frontend_method_data,
        frontend=frontend,
        on_device=on_device,
    )


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", prune_function=False),
        num_arrays=3,
        min_value=-1e3,
        max_value=1e3,
    ).filter(lambda x: all(dt == "float32" for dt in x[0])),
)
def test_torch_instance_backward(
    dtype_x,
):
    if ivy.current_backend_str() == "numpy":
        ivy.warnings.warn("Gradient calculation unavailable for numpy backend")
        return
    if ivy.current_backend_str() == "paddle":
        ivy.warnings.warn("torch.Tensor.backward() unavailable for paddle backend")
        return
    _, values = dtype_x
    x = Tensor(values[0], requires_grad=True)
    y = Tensor(values[1], requires_grad=True)
    z = Tensor(values[2], requires_grad=True)
    a = x + y.pow(2)
    b = z * a
    c = b.sum()
    c.backward()
    x_torch = torch.tensor(values[0], requires_grad=True, dtype=torch.float32)
    y_torch = torch.tensor(values[1], requires_grad=True, dtype=torch.float32)
    z_torch = torch.tensor(values[2], requires_grad=True, dtype=torch.float32)
    a_torch = x_torch + y_torch.pow(2)
    b_torch = z_torch * a_torch
    c_torch = b_torch.sum()
    c_torch.backward()
    helpers.assertions.value_test(
        ret_np_flat=helpers.flatten_and_to_np(ret=x._grads.ivy_array),
        ret_np_from_gt_flat=helpers.flatten_and_to_np(
            ret=ivy.to_ivy(x_torch.grad.numpy())
        ),
        rtol=1e-3,
        atol=1e-3,
        ground_truth_backend="torch",
    )
    helpers.assertions.value_test(
        ret_np_flat=helpers.flatten_and_to_np(ret=y._grads.ivy_array),
        ret_np_from_gt_flat=helpers.flatten_and_to_np(
            ret=ivy.to_ivy(y_torch.grad.numpy())
        ),
        rtol=1e-3,
        atol=1e-3,
        ground_truth_backend="torch",
    )
    helpers.assertions.value_test(
        ret_np_flat=helpers.flatten_and_to_np(ret=z._grads.ivy_array),
        ret_np_from_gt_flat=helpers.flatten_and_to_np(
            ret=ivy.to_ivy(z_torch.grad.numpy())
        ),
        rtol=1e-3,
        atol=1e-3,
        ground_truth_backend="torch",
    )
