# global
from hypothesis import given, strategies as st, assume
import numpy as np

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_method
from ivy.functional.frontends.jax import DeviceArray


CLASS_TREE = "ivy.functional.frontends.jax.DeviceArray"


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False)
    ),
)
def test_jax_devicearray_property_ivy_array(
    dtype_x,
):
    _, data = dtype_x
    x = DeviceArray(data[0])
    ret = helpers.flatten_and_to_np(ret=x.ivy_array.data)
    ret_gt = helpers.flatten_and_to_np(ret=data[0])
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend="jax",
    )


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False)
    ),
)
def test_jax_devicearray_property_dtype(
    dtype_x,
):
    dtype, data = dtype_x
    x = DeviceArray(data[0])
    assert x.dtype == dtype[0]


@given(
    dtype_x_shape=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False),
        ret_shape=True,
    ),
)
def test_jax_devicearray_property_shape(
    dtype_x_shape,
):
    _, data, shape = dtype_x_shape
    x = DeviceArray(data[0])
    assert x.shape == shape


@st.composite
def _at_helper(draw):
    _, data, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid", prune_function=False),
            num_arrays=2,
            shared_dtype=True,
            min_num_dims=1,
            ret_shape=True,
        )
    )
    axis = draw(helpers.get_axis(shape=shape, force_tuple=True))
    index = ()
    for a in axis:
        index = index + (draw(st.integers(min_value=0, max_value=shape[a] - 1)),)
    return data, index


@given(
    x_y_index=_at_helper(),
)
def test_jax_devicearray_property_at(x_y_index):
    xy, idx = x_y_index
    x = DeviceArray(xy[0])
    y = DeviceArray(xy[1])
    idx = idx[0]
    x_set = x.at[idx].set(y[idx])
    assert x_set[idx] == y[idx]
    assert x.at[idx].get() == x[idx]


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="jax.numpy.array",
    method_name="all",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        force_int_axis=True,
        valid_axis=True,
        min_num_dims=1,
    ),
    keepdims=st.booleans(),
)
def test_jax_devicearray_all(
    dtype_x_axis,
    keepdims,
    on_device,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "axis": axis,
            "keepdims": keepdims,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="jax.numpy.array",
    method_name="argmax",
    dtype_and_x=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        force_int_axis=True,
        min_num_dims=1,
        valid_axis=True,
    ),
    keepdims=st.booleans(),
)
def test_jax_devicearray_argmax(
    dtype_and_x,
    keepdims,
    on_device,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
):
    input_dtype, x, axis = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "axis": axis,
            "keepdims": keepdims,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="jax.numpy.array",
    method_name="conj",
    dtype_and_x=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("real_and_complex"),
    ),
)
def test_jax_devicearray_conj(
    dtype_and_x,
    on_device,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
):
    input_dtype, x, axis = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "object": x[0],
        },
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
    init_tree="jax.numpy.array",
    method_name="mean",
    dtype_and_x=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        force_int_axis=True,
        min_num_dims=1,
        valid_axis=True,
    ),
    keepdims=st.booleans(),
)
def test_jax_devicearray_mean(
    dtype_and_x,
    keepdims,
    on_device,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
):
    input_dtype, x, axis = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "axis": axis,
            "keepdims": keepdims,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="jax.numpy.array",
    method_name="cumprod",
    dtype_and_x=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=5,
        min_value=-100,
        max_value=100,
        valid_axis=True,
        allow_neg_axes=False,
        max_axes_size=1,
        force_int_axis=True,
    ),
)
def test_jax_devicearray_cumprod(
    dtype_and_x,
    on_device,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
):
    input_dtype, x, axis = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "axis": axis,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="jax.numpy.array",
    method_name="nonzero",
    dtype_and_x=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
    ),
)
def test_jax_devicearray_nonzero(
    dtype_and_x,
    on_device,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
):
    input_dtype, x, axis = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "object": x[0],
        },
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
    init_tree="jax.numpy.array",
    method_name="ravel",
    dtype_and_x=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=2,
        max_dim_size=10,
        shape=helpers.get_shape(
            min_num_dims=2, max_num_dims=5, min_dim_size=2, max_dim_size=10
        ),
    ),
    order=st.sampled_from(["C", "F"]),
)
def test_jax_devicearray_ravel(
    dtype_and_x,
    order,
    on_device,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
):
    input_dtype, x, axis = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "order": order,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="jax.numpy.array",
    method_name="__pos__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_jax_devicearray__pos_(
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
        on_device=on_device,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="jax.numpy.array",
    method_name="__neg__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("signed_integer"),
    ),
)
def test_jax_devicearray__neg_(
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
            "object": x[0],
        },
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
    init_tree="jax.numpy.array",
    method_name="__eq__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
)
def test_jax_devicearray__eq_(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_and_x
    assume("bfloat16" not in input_dtype)
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="jax.numpy.array",
    method_name="__ne__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
)
def test_jax_devicearray__ne_(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_and_x
    assume("bfloat16" not in input_dtype)
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="jax.numpy.array",
    method_name="__lt__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_integer"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_jax_devicearray__lt_(
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
            "object": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="jax.numpy.array",
    method_name="__le__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_integer"),
        num_arrays=2,
    ),
)
def test_jax_devicearray__le_(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_and_x
    assume("bfloat16" not in input_dtype)
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="jax.numpy.array",
    method_name="__gt__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_integer"),
        num_arrays=2,
    ),
)
def test_jax_devicearray__gt_(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_and_x
    assume("bfloat16" not in input_dtype)
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="jax.numpy.array",
    method_name="__ge__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_integer"),
        num_arrays=2,
    ),
)
def test_jax_devicearray__ge_(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_and_x
    assume("bfloat16" not in input_dtype)
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="jax.numpy.array",
    method_name="__abs__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
)
def test_jax_devicearray__abs_(
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
            "object": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@st.composite
def _get_dtype_x_and_int(draw, *, dtype="numeric"):
    x_dtype, x = draw(
        helpers.dtype_and_values(available_dtypes=helpers.get_dtypes(dtype))
    )
    pow_dtype, x_int = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("integer"),
            min_value=0,
            max_value=10,
            max_num_dims=0,
            max_dim_size=1,
        )
    )
    x_dtype = x_dtype + pow_dtype
    return x_dtype, x, x_int


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="jax.numpy.array",
    method_name="__pow__",
    dtype_x_pow=_get_dtype_x_and_int(),
)
def test_jax_devicearray__pow_(
    dtype_x_pow,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x, pow = dtype_x_pow
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": pow[0],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="jax.numpy.array",
    method_name="__rpow__",
    dtype_x_pow=_get_dtype_x_and_int(),
)
def test_jax_devicearray__rpow_(
    dtype_x_pow,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x, pow = dtype_x_pow
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "object": pow[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[0],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="jax.numpy.array",
    method_name="__and__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_jax_devicearray__and_(
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
            "object": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"other": x[1]},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="jax.numpy.array",
    method_name="__rand__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_jax_devicearray__rand_(
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
            "object": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"other": x[1]},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="jax.numpy.array",
    method_name="__or__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_jax_devicearray__or_(
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
            "object": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"other": x[1]},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="jax.numpy.array",
    method_name="__ror__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_jax_devicearray__ror_(
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
            "object": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"other": x[1]},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="jax.numpy.array",
    method_name="__xor__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_jax_devicearray__xor_(
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
            "object": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"other": x[1]},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="jax.numpy.array",
    method_name="__rxor__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_jax_devicearray__rxor_(
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
            "object": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"other": x[1]},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="jax.numpy.array",
    method_name="__invert__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
    ),
)
def test_jax_devicearray__invert_(
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
            "object": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# shifting helper
@st.composite
def _get_dtype_x_and_int_shift(draw, dtype):
    x_dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes(dtype),
            num_arrays=2,
            shared_dtype=True,
        )
    )
    x_dtype = x_dtype
    x[1] = np.asarray(np.clip(x[0], 0, np.iinfo(x_dtype[0]).bits - 1), dtype=x_dtype[0])
    return x_dtype, x[0], x[1]


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="jax.numpy.array",
    method_name="__lshift__",
    dtype_x_shift=_get_dtype_x_and_int_shift(dtype="signed_integer"),
)
def test_jax_special_lshift(
    dtype_x_shift,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x, shift = dtype_x_shift
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "object": x,
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"other": shift},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="jax.numpy.array",
    method_name="__rlshift__",
    dtype_x_shift=_get_dtype_x_and_int_shift(dtype="signed_integer"),
)
def test_jax_special_rlshift(
    dtype_x_shift,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x, shift = dtype_x_shift
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "object": shift,
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"other": x},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="jax.numpy.array",
    method_name="__rshift__",
    dtype_x_shift=_get_dtype_x_and_int_shift(dtype="signed_integer"),
)
def test_jax_special_rshift(
    dtype_x_shift,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x, shift = dtype_x_shift
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "object": x,
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"other": shift},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="jax.numpy.array",
    method_name="__rrshift__",
    dtype_x_shift=_get_dtype_x_and_int_shift(dtype="signed_integer"),
)
def test_jax_special_rrshift(
    dtype_x_shift,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x, shift = dtype_x_shift
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "object": shift,
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"other": x},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="jax.numpy.array",
    method_name="__add__",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
        shared_dtype=True,
        num_arrays=2,
    ),
)
def test_jax_special_add(
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
            "object": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"other": x[1]},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="jax.numpy.array",
    method_name="__radd__",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
        shared_dtype=True,
        num_arrays=2,
    ),
)
def test_jax_special_radd(
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
            "object": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"other": x[1]},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="jax.numpy.array",
    method_name="__sub__",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
        shared_dtype=True,
        num_arrays=2,
    ),
)
def test_jax_special_sub(
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
            "object": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"other": x[1]},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="jax.numpy.array",
    method_name="__rsub__",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
        shared_dtype=True,
        num_arrays=2,
    ),
)
def test_jax_special_rsub(
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
            "object": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"other": x[1]},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="jax.numpy.array",
    method_name="__mul__",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
        shared_dtype=True,
        num_arrays=2,
    ),
)
def test_jax_special_mul(
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
            "object": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"other": x[1]},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="jax.numpy.array",
    method_name="__rmul__",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
        shared_dtype=True,
        num_arrays=2,
    ),
)
def test_jax_special_rmul(
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
            "object": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"other": x[1]},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="jax.numpy.array",
    method_name="__div__",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
        shared_dtype=True,
        num_arrays=2,
    ),
)
def test_jax_special_div(
    dtype_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_x
    assume(not np.any(np.isclose(x[1], 0)))
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"other": x[1]},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="jax.numpy.array",
    method_name="__rdiv__",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
        shared_dtype=True,
        num_arrays=2,
    ),
)
def test_jax_special_rdiv(
    dtype_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_x
    assume(not np.any(np.isclose(x[0], 0)))
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"other": x[1]},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="jax.numpy.array",
    method_name="__truediv__",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
        shared_dtype=True,
        num_arrays=2,
    ),
)
def test_jax_special_truediv(
    dtype_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_x
    assume(not np.any(np.isclose(x[1], 0)))
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"other": x[1]},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="jax.numpy.array",
    method_name="__rtruediv__",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
        shared_dtype=True,
        num_arrays=2,
    ),
)
def test_jax_special_rtruediv(
    dtype_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_x
    assume(not np.any(np.isclose(x[0], 0)))
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"other": x[1]},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="jax.numpy.array",
    method_name="__mod__",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
        shared_dtype=True,
        num_arrays=2,
    ),
)
def test_jax_special_mod(
    dtype_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_x
    assume(not np.any(np.isclose(x[1], 0)))
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"other": x[1]},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="jax.numpy.array",
    method_name="__rmod__",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
        shared_dtype=True,
        num_arrays=2,
    ),
)
def test_jax_special_rmod(
    dtype_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_x
    assume(not np.any(np.isclose(x[0], 0)))
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"other": x[1]},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@st.composite
def _get_dtype_input_and_vectors(draw):
    dim_size = draw(helpers.ints(min_value=2, max_value=5))
    dtype = draw(helpers.get_dtypes("numeric", index=1, full=False))
    vec1 = draw(
        helpers.array_values(
            dtype=dtype[0], shape=(dim_size, dim_size), min_value=2, max_value=5
        )
    )
    vec2 = draw(
        helpers.array_values(
            dtype=dtype[0], shape=(dim_size, dim_size), min_value=2, max_value=5
        )
    )
    return dtype, [vec1, vec2]


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="jax.numpy.array",
    method_name="__matmul__",
    dtype_x=_get_dtype_input_and_vectors(),
)
def test_jax_special_matmul(
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
            "object": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"other": x[1]},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="jax.numpy.array",
    method_name="__rmatmul__",
    dtype_x=_get_dtype_input_and_vectors(),
)
def test_jax_special_rmatmul(
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
            "object": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"other": x[1]},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# __getitem__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="jax.numpy.array",
    method_name="__getitem__",
    dtype_x_index=helpers.dtype_array_index(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_jax_special_getitem(
    dtype_x_index,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x, index = dtype_x_index
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtype[0]],
        init_all_as_kwargs_np={"object": x},
        method_input_dtypes=[input_dtype[1]],
        method_all_as_kwargs_np={"idx": index},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )
