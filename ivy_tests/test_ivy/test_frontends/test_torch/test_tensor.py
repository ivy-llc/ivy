# global
import pytest
from types import SimpleNamespace

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
)
from ivy.functional.frontends.torch import Tensor
from ivy_tests.test_ivy.helpers import handle_frontend_method
from ivy_tests.test_ivy.test_functional.test_core.test_manipulation import _get_splits
from ivy_tests.test_ivy.test_functional.test_core.test_searching import (
    _broadcastable_trio,
)
from ivy_tests.test_ivy.test_functional.test_experimental.test_core.test_manipulation import (  # noqa
    _get_split_locations,
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
    ivy.utils.assertions.check_equal(x.device, ivy.dev(ivy.array(data[0])))


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
    ivy.utils.assertions.check_equal(x.dtype, dtype[0])


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False),
        ret_shape=True,
    ).filter(lambda x: "bfloat16" not in x[0]),
)
def test_torch_tensor_property_shape(dtype_x):
    dtype, data, shape = dtype_x
    x = Tensor(data[0])
    ivy.utils.assertions.check_equal(x.ivy_array.shape, ivy.Shape(shape))


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
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="reshape",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", full=True),
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
    )


# reshape_as
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="reshape_as",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", full=True), num_arrays=2
    ),
)
def test_torch_instance_reshape_as(
    dtype_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
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
    )


# sum
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="sum",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-1e04,
        max_value=1e04,
    ),
)
def test_torch_instance_sum(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=["float64"] + input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=["float64"] + input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
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
    )


# view
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="view",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", full=True),
        shape=st.shared(helpers.get_shape(), key="value_shape"),
    ),
    shape=helpers.reshape_shapes(
        shape=st.shared(helpers.get_shape(), key="value_shape")
    ),
)
def test_torch_instance_view(
    dtype_x,
    shape,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "shape": shape,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="float",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", full=True),
    ),
)
def test_torch_instance_float(
    dtype_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
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
    )


# amax
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="amax",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
    ),
)
def test_torch_instance_amax(
    dtype_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
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
    )


# amin
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="amin",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
    ),
)
def test_torch_instance_amin(
    dtype_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
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
    )


# aminmax
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="aminmax",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
    ),
    keepdim=st.booleans(),
)
def test_torch_instance_aminmax(
    dtype_x,
    keepdim,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "keepdim": keepdim,
        },
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
    )


@st.composite
def _to_helper(draw):
    dtype_x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            num_arrays=2,
            large_abs_safety_factor=2,
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
    )


@st.composite
def _array_and_index(
    draw,
    *,
    available_dtypes=helpers.get_dtypes("numeric"),
    min_num_dims=1,
    max_num_dims=3,
    min_dim_size=1,
    max_dim_size=10,
    shape=None,
):
    if isinstance(min_dim_size, st._internal.SearchStrategy):
        min_dim_size = draw(min_dim_size)
    if isinstance(max_dim_size, st._internal.SearchStrategy):
        max_dim_size = draw(max_dim_size)
    if isinstance(available_dtypes, st._internal.SearchStrategy):
        available_dtypes = draw(available_dtypes)

    assert available_dtypes is not None, "Unspecified dtype or available_dtypes."
    dtype = draw(
        helpers.array_dtypes(
            num_arrays=1,
            available_dtypes=available_dtypes,
        )
    )
    dtype.append("int32")

    if shape is not None:
        if not isinstance(shape, (tuple, list)):
            shape = draw(shape)
    else:
        shape = draw(
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

    array = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=shape,
        )
    )

    index = tuple([draw(helpers.ints(min_value=0, max_value=_ - 1)) for _ in shape])
    index = index if len(index) != 0 else index[0]
    return dtype, [array, index]


# __getitem__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__getitem__",
    dtype_and_x=_array_and_index(available_dtypes=helpers.get_dtypes("numeric")),
)
def test_torch_instance_getitem(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
):
    input_dtype, x = dtype_and_x
    data = x[0]
    index = x[1]
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtype[0]],
        init_all_as_kwargs_np={"data": data},
        method_input_dtypes=[input_dtype[1]],
        method_all_as_kwargs_np={"query": index},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
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
    split_size=_get_splits().filter(lambda s: s is not None),
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
    indices_or_sections=_get_split_locations(min_num_dims=2, axis=0),
)
def test_torch_instance_vsplit(
    dtype_value,
    indices_or_sections,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
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
    )


# hsplit
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="hsplit",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
    ),
    indices_or_sections=_get_split_locations(min_num_dims=1, axis=1),
)
def test_torch_instance_hsplit(
    dtype_value,
    indices_or_sections,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
):
    input_dtype, x = dtype_value
    # TODO: remove the assumption when these bugfixes are merged and version-pinned
    # https://github.com/tensorflow/tensorflow/pull/59523
    # https://github.com/google/jax/pull/14275
    assume(
        not (
            len(x[0].shape) == 1 and ivy.current_backend_str() in ("tensorflow", "jax")
        )
    )
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
    indices_or_sections=_get_split_locations(min_num_dims=3, axis=2),
)
def test_torch_instance_dsplit(
    dtype_value,
    indices_or_sections,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
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
    )


# expand_as
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="expand_as",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", full=True), num_arrays=2
    ),
)
def test_torch_instance_expand_as(
    dtype_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
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
    )


# max
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="max",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
    ),
)
def test_torch_instance_max(
    dtype_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
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
    )


# is_cuda
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="is_cuda",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
    size=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtypes=_dtypes(),
    requires_grad=_requires_grad(),
    device=st.booleans(),
)
def test_torch_instance_is_cuda(
    dtype_and_x,
    size,
    dtypes,
    requires_grad,
    device,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
):
    input_dtype, x = dtype_and_x
    device = "cpu" if device is False else "gpu:0"
    x = Tensor(x[0]).new_ones(
        size=size, dtype=dtypes[0], device=device, requires_grad=requires_grad
    )

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x,
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
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
    )


# bitwise_or
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="bitwise_or",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
    ),
)
def test_torch_instance_bitwise_or(
    dtype_and_x,
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
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
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
    )


# argsort
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="argsort",
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=3,
        min_axis=-1,
        max_axis=0,
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
    )


# min
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="min",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
    ),
)
def test_torch_instance_min(
    dtype_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
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
    unpack_dims=st.booleans(),
)
def test_torch_instance_permute(
    dtype_values_axis,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
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
    )


# mean
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="mean",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-1e04,
        max_value=1e04,
    ),
)
def test_torch_instance_mean(
    dtype_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
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
    start_dim=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="shape"),
        allow_neg=True,
        force_int=True,
    ),
    end_dim=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="shape"),
        allow_neg=True,
        force_int=True,
    ),
)
def test_torch_instance_flatten(
    dtype_value,
    start_dim,
    end_dim,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
):
    if start_dim > end_dim:
        temp = start_dim
        start_dim = end_dim
        end_dim = temp
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "start_dim": start_dim,
            "end_dim": end_dim,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
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
    )


# squeeze
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="squeeze",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ),
    dim=helpers.get_shape(min_num_dims=1),
)
def test_torch_instance_squeeze(
    dtype_value,
    dim,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
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
    )


# real
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="real",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_real(
    dtype_and_x,
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
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
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
    )
