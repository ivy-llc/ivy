# global
import numpy as np
from hypothesis import assume, strategies as st, given

# local
import ivy
from ivy.functional.frontends.numpy import ndarray
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import (
    handle_frontend_method,
    assert_all_close,
    update_backend,
)
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.test_functional.test_core.test_linalg import (
    _get_first_matrix_and_dtype,
    _get_second_matrix_and_dtype,
)

# from ivy.functional.frontends.numpy import ndarray
from ivy_tests.test_ivy.test_frontends.test_numpy.test_mathematical_functions.test_miscellaneous import (  # noqa
    _get_clip_inputs,
)
from ivy_tests.test_ivy.test_frontends.test_numpy.test_mathematical_functions.test_sums_products_differences import (  # noqa
    _get_castable_dtypes_values,
)


CLASS_TREE = "ivy.functional.frontends.numpy.ndarray"


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False),
        ret_shape=True,
    ),
)
def test_numpy_ivy_array(
    dtype_x,
    frontend,
    backend_fw,
):
    dtype, data, shape = dtype_x
    with update_backend(backend_fw) as ivy_backend:
        x = ivy_backend.functional.frontends.numpy.ndarray(shape, dtype[0])
        x.ivy_array = data[0]
        ret = helpers.flatten_and_to_np(ret=x.ivy_array.data, backend=backend_fw)
        ret_gt = helpers.flatten_and_to_np(ret=data[0], backend=frontend)
        helpers.value_test(
            ret_np_flat=ret,
            ret_np_from_gt_flat=ret_gt,
            backend=backend_fw,
            ground_truth_backend="numpy",
        )


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False),
        ret_shape=True,
    ),
)
def test_numpy_dtype(dtype_x, backend_fw, frontend):
    dtype, data, shape = dtype_x
    with update_backend(backend_fw) as ivy_backend:
        x = ivy_backend.functional.frontends.numpy.ndarray(shape, dtype[0])
        x.ivy_array = data[0]
        ivy_backend.utils.assertions.check_equal(
            x.dtype, ivy.Dtype(dtype[0]), as_array=False
        )


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False),
        ret_shape=True,
    ),
)
def test_numpy_shape(
    dtype_x,
    backend_fw,
):
    dtype, data, shape = dtype_x
    with update_backend(backend_fw) as ivy_backend:
        x = ivy_backend.functional.frontends.numpy.ndarray(shape, dtype[0])
        x.ivy_array = data[0]
        ivy_backend.utils.assertions.check_equal(
            x.shape, ivy.Shape(shape), as_array=False
        )


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False),
        ret_shape=True,
    ),
)
def test_numpy_ndarray_property_ndim(dtype_x, backend_fw):
    dtype, data, shape = dtype_x
    with update_backend(backend_fw) as ivy_backend:
        x = ivy_backend.functional.frontends.numpy.ndarray(shape, dtype[0])
        x.ivy_array = data[0]
        ivy_backend.utils.assertions.check_equal(x.ndim, data[0].ndim, as_array=False)


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False),
        ret_shape=True,
    ),
)
def test_numpy_size(
    dtype_x,
):
    dtype, data, shape = dtype_x
    x = ndarray(shape, dtype[0])
    x.ivy_array = data[0]
    ivy.utils.assertions.check_equal(x.size, data[0].size, as_array=False)


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False),
        ret_shape=True,
    ),
)
def test_numpy_T(
    dtype_x,
    backend_fw,
    frontend,
):
    dtype, data, shape = dtype_x
    with update_backend(backend_fw) as ivy_backend:
        x = ivy_backend.functional.frontends.numpy.ndarray(shape, dtype[0])
        x.ivy_array = data[0]
        ret = helpers.flatten_and_to_np(ret=x.T.ivy_array, backend=backend_fw)
        ret_gt = helpers.flatten_and_to_np(
            ret=ivy_backend.permute_dims(
                ivy_backend.native_array(data[0]), list(range(len(shape)))[::-1]
            ),
            backend=backend_fw,
        )
        helpers.value_test(
            ret_np_flat=ret,
            ret_np_from_gt_flat=ret_gt,
            backend=backend_fw,
            ground_truth_backend="numpy",
        )


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", prune_function=False),
        num_arrays=1,
        ret_shape=True,
    )
)
def test_numpy_flat(dtype_x, backend_fw):
    dtype, data, shape = dtype_x

    with update_backend(backend_fw) as ivy_backend:
        x = ivy_backend.functional.frontends.numpy.ndarray(shape, dtype[0])
        x.ivy_array = data[0]

        flat_ivy = x.flat
        flat_ivy = flat_ivy.ivy_array.to_numpy()
        flat_generated = ivy_backend.to_numpy(data[0]).flatten()
        ivy_backend.utils.assertions.check_equal(
            flat_ivy, flat_generated, as_array=True
        )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="astype",
    dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(
        arr_func=[
            lambda: helpers.dtype_and_values(
                available_dtypes=helpers.get_dtypes("numeric"),
            )
        ],
    ),
    order=st.sampled_from(["C", "F", "A", "K"]),
    copy=st.booleans(),
)
def test_numpy_astype(
    dtypes_values_casting,
    order,
    copy,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, x, casting, dtype = dtypes_values_casting
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "dtype": dtype if dtype else input_dtypes[0],
            "order": order,
            "casting": casting,
            "copy": copy,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="argmax",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=st.one_of(
            helpers.get_dtypes("numeric"),
        ),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
    keep_dims=st.booleans(),
)
def test_numpy_argmax(
    dtype_x_axis,
    keep_dims,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, x, axis = dtype_x_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "axis": axis,
            "keepdims": keep_dims,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@st.composite
def dtypes_x_reshape(draw):
    dtypes, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            shape=helpers.get_shape(
                allow_none=False,
                min_num_dims=1,
                max_num_dims=5,
                min_dim_size=1,
                max_dim_size=10,
            ),
        )
    )
    shape = draw(helpers.reshape_shapes(shape=np.array(x).shape))
    return dtypes, x, shape


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="reshape",
    dtypes_x_shape=dtypes_x_reshape(),
    order=st.sampled_from(["C", "F", "A"]),
)
def test_numpy_reshape(
    dtypes_x_shape,
    order,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, x, shape = dtypes_x_shape
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={
            "newshape": shape,
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
    init_tree="numpy.array",
    method_name="transpose",
    array_and_axes=np_frontend_helpers._array_and_axes_permute_helper(
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=2,
        max_dim_size=10,
    ),
)
def test_numpy_transpose(
    array_and_axes,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    array, input_dtypes, axes = array_and_axes
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": np.array(array),
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "axes": axes,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# swapaxes
@st.composite
def dtype_values_and_axes(draw):
    dtype, x, x_shape = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"),
            min_num_dims=1,
            max_num_dims=5,
            ret_shape=True,
        )
    )
    axis1, axis2 = draw(
        helpers.get_axis(
            shape=x_shape,
            sort_values=False,
            unique=True,
            min_size=2,
            max_size=2,
            force_tuple=True,
        )
    )
    return dtype, x, axis1, axis2


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="swapaxes",
    dtype_x_and_axes=dtype_values_and_axes(),
)
def test_numpy_swapaxes(
    dtype_x_and_axes,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    on_device,
):
    input_dtypes, x, axis1, axis2 = dtype_x_and_axes
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        method_input_dtypes=input_dtypes,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_all_as_kwargs_np={
            "axis1": axis1,
            "axis2": axis2,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# any
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="any",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        valid_axis=True,
        force_int_axis=True,
        allow_neg_axes=True,
    ),
    keepdims=st.booleans(),
    where=np_frontend_helpers.where(),
)
def test_numpy_any(
    dtype_x_axis,
    keepdims,
    where,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, x, axis = dtype_x_axis
    (
        where,
        input_dtypes,
        method_flags,
    ) = np_frontend_helpers.handle_where_and_array_bools(
        where=[where[0][0]] if isinstance(where, list) else where,
        input_dtype=input_dtypes,
        test_flags=method_flags,
    )

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "axis": axis,
            "out": None,
            "keepdims": keepdims,
            "where": where,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="all",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid", full=True),
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        valid_axis=True,
        force_int_axis=True,
        allow_neg_axes=True,
    ),
    keepdims=st.booleans(),
    where=np_frontend_helpers.where(),
)
def test_numpy_all(
    dtype_x_axis,
    keepdims,
    where,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, x, axis = dtype_x_axis
    (
        where,
        input_dtypes,
        method_flags,
    ) = np_frontend_helpers.handle_where_and_array_bools(
        where=[where[0][0]] if isinstance(where, list) else where,
        input_dtype=input_dtypes,
        test_flags=method_flags,
    )

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "axis": axis,
            "out": None,
            "keepdims": keepdims,
            "where": where,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="argsort",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
)
def test_numpy_argsort(
    dtype_x_axis,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, x, axis = dtype_x_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        method_input_dtypes=input_dtypes,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        method_all_as_kwargs_np={
            "axis": axis,
            "kind": None,
            "order": None,
        },
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="mean",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
)
def test_numpy_mean(
    dtype_x_axis,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, x, axis = dtype_x_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        method_input_dtypes=input_dtypes,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_all_as_kwargs_np={
            "axis": axis,
            "dtype": "float64",
            "out": None,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        rtol_=1e-2,
        atol_=1e-2,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="min",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
    keepdims=st.booleans(),
)
def test_numpy_min(
    dtype_x_axis,
    keepdims,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, x, axis = dtype_x_axis

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtypes,
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


# prod
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="prod",
    dtype_x_axis_dtype=_get_castable_dtypes_values(use_where=True),
    keep_dims=st.booleans(),
    initial=st.one_of(st.floats(min_value=-100, max_value=100)),
)
def test_numpy_prod(
    dtype_x_axis_dtype,
    keep_dims,
    initial,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, x, axis, dtype, where = dtype_x_axis_dtype
    if ivy.current_backend_str() == "torch":
        assume(not method_flags.as_variable[0])

    (
        where,
        input_dtypes,
        method_flags,
    ) = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=method_flags,
    )
    where = ivy.array(where, dtype="bool")
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "axis": axis,
            "dtype": dtype,
            "keepdims": keep_dims,
            "initial": initial,
            "where": where,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="argmin",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
    ),
    keepdims=st.booleans(),
)
def test_numpy_argmin(
    dtype_x_axis,
    keepdims,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, x, axis = dtype_x_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtypes,
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
    init_tree="numpy.array",
    method_name="clip",
    input_and_ranges=_get_clip_inputs(),
)
def test_numpy_clip(
    input_and_ranges,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, x, min, max = input_and_ranges
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        method_input_dtypes=input_dtypes,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_all_as_kwargs_np={
            "min": min,
            "max": max,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="compress",
    dtype_arr_ax=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=10,
        max_dim_size=100,
        valid_axis=True,
        force_int_axis=True,
    ),
    condition=helpers.array_values(
        dtype=helpers.get_dtypes("bool"),
        shape=helpers.get_shape(
            min_num_dims=1, max_num_dims=1, min_dim_size=1, max_dim_size=5
        ),
    ),
)
def test_numpy_ndarray_compress(
    dtype_arr_ax,
    condition,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtypes, arr, ax = dtype_arr_ax
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        init_all_as_kwargs_np={
            "object": arr[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "condition": condition,
            "axis": ax,
            "out": None,
        },
        frontend=frontend,
        backend_to_test=backend_fw,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="conj",
    dtype_and_x=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("real_and_complex"),
    ),
)
def test_numpy_conj(
    dtype_and_x,
    on_device,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
):
    input_dtype, x, axis = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
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
    init_tree="numpy.array",
    method_name="max",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
    keepdims=st.booleans(),
)
def test_numpy_max(
    dtype_x_axis,
    keepdims,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, x, axis = dtype_x_axis

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtypes,
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
    init_tree="numpy.array",
    method_name="cumprod",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
    dtype=helpers.get_dtypes("float", full=False, none=True),
)
def test_numpy_cumprod(
    dtype_x_axis,
    dtype,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, x, axis = dtype_x_axis

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "axis": axis,
            "dtype": dtype[0],
            "out": None,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="cumsum",
    dtype_x_axis_dtype=_get_castable_dtypes_values(),
)
def test_numpy_cumsum(
    dtype_x_axis_dtype,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, x, axis, dtype = dtype_x_axis_dtype
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "axis": axis,
            "dtype": dtype,
            "out": None,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="dot",
    dtype_and_x=np_frontend_helpers._get_dtype_input_and_vectors(),
)
def test_numpy_instance_dot(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtype, x, other = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": x,
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "b": other,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="diagonal",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_axis=0,
        max_axis=1,
        min_num_dims=1,
        force_int_axis=True,
    ),
)
def test_numpy_diagonal(
    dtype_x_axis,
    offset,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, x, axis1, axis2 = dtype_x_axis

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "axis1": axis1,
            "axis2": axis2,
            "offset": offset,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="sort",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
)
def test_numpy_sort(
    dtype_x_axis,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, x, axis = dtype_x_axis

    ret, frontend_ret = helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "axis": axis,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        test_values=False,
        on_device=on_device,
    )
    frontend_ret = np.sort(x[0], axis=axis)
    assert_all_close(
        ret_np=ret,
        ret_from_gt_np=frontend_ret,
        rtol=1e-2,
        atol=1e-2,
        backend=backend_fw,
        ground_truth_backend="numpy",
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="copy",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
    ),
)
def test_numpy_copy(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, x = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="nonzero",
    dtype_and_a=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_numpy_nonzero(
    dtype_and_a,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, a = dtype_and_a

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        init_all_as_kwargs_np={
            "object": a[0],
        },
        method_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        method_all_as_kwargs_np={},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="ravel",
    dtype_and_a=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_numpy_ravel(
    dtype_and_a,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, a = dtype_and_a

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        init_all_as_kwargs_np={
            "object": a[0],
        },
        method_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        method_all_as_kwargs_np={},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="repeat",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=2,
        min_dim_size=2,
    ),
    repeats=helpers.ints(min_value=2, max_value=5),
    axis=helpers.ints(min_value=-1, max_value=1),
)
def test_numpy_repeat(
    dtype_and_x,
    repeats,
    axis,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, x = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "repeats": repeats,
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
    init_tree="numpy.array",
    method_name="searchsorted",
    dtype_x_v=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("signed_integer"),
        min_num_dims=1,
        max_num_dims=1,
        num_arrays=2,
    ),
    side=st.sampled_from(["left", "right"]),
)
def test_numpy_searchsorted(
    dtype_x_v,
    side,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, xs = dtype_x_v

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        init_all_as_kwargs_np={
            "object": xs[0],
        },
        method_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        method_all_as_kwargs_np={
            "v": xs[1],
            "side": side,
            "sorter": np.argsort(xs[0]),
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="squeeze",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
)
def test_numpy_squeeze(
    dtype_x_axis,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, x, axis = dtype_x_axis

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtypes,
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
    init_tree="numpy.array",
    method_name="std",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        max_value=100,
        valid_axis=True,
        force_int_axis=True,
    ),
    keepdims=st.booleans(),
    where=np_frontend_helpers.where(),
)
def test_numpy_std(
    dtype_x_axis,
    keepdims,
    where,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, x, axis = dtype_x_axis
    (
        where,
        input_dtypes,
        method_flags,
    ) = np_frontend_helpers.handle_where_and_array_bools(
        where=[where[0][0]] if isinstance(where, list) else where,
        input_dtype=input_dtypes,
        test_flags=method_flags,
    )
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        method_input_dtypes=input_dtypes,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_all_as_kwargs_np={
            "axis": axis,
            "out": None,
            "ddof": 0,
            "keepdims": keepdims,
            "where": where,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# fill
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="fill",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric")
    ),
)
def test_numpy_fill(
    dtype_and_x,
    num,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={
            "num": num,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="__add__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"), num_arrays=2
    ),
)
def test_numpy___add__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, xs = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": xs[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "value": xs[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="__radd__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"), num_arrays=2
    ),
)
def test_numpy___radd__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, xs = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": xs[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "value": xs[1],
        },
        method_flags=method_flags,
        init_flags=init_flags,
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="__sub__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"), num_arrays=2
    ),
)
def test_numpy___sub__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, xs = dtype_and_x

    helpers.test_frontend_method(
        backend_to_test=backend_fw,
        init_input_dtypes=input_dtypes,
        init_all_as_kwargs_np={
            "object": xs[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "value": xs[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="__mul__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
)
def test_numpy___mul__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, xs = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": xs[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "value": xs[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="__rmul__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
)
def test_numpy___rmul__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, xs = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": xs[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "value": xs[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# __floordiv__ test
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="__floordiv__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        allow_inf=False,
        large_abs_safety_factor=4,
        safety_factor_scale="linear",
        shared_dtype=True,
    ),
)
def test_numpy___floordiv__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, xs = dtype_and_x
    assume(not np.any(np.isclose(xs[1], 0)))
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": xs[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "value": xs[1],
        },
        init_flags=init_flags,
        method_flags=method_flags,
        frontend_method_data=frontend_method_data,
        frontend=frontend,
        atol_=1,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="__truediv__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
)
def test_numpy___truediv__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, xs = dtype_and_x
    assume(not np.any(np.isclose(xs[0], 0)))

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": xs[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "value": xs[1],
        },
        init_flags=init_flags,
        method_flags=method_flags,
        frontend_method_data=frontend_method_data,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="__rtruediv__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
)
def test_numpy___rtruediv__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, xs = dtype_and_x
    assume(not np.any(np.isclose(xs[0], 0)))

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": xs[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "value": xs[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="__pow__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    power=helpers.ints(min_value=1, max_value=3),
)
def test_numpy___pow__(
    dtype_and_x,
    power,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, xs = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": xs[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "value": power,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="__and__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("bool"),
        num_arrays=2,
    ),
)
def test_numpy___and__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, xs = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": xs[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "value": xs[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="__or__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("bool"),
        num_arrays=2,
    ),
)
def test_numpy___or__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, xs = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": xs[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "value": xs[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="__xor__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("bool"),
        num_arrays=2,
    ),
)
def test_numpy___xor__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, xs = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": xs[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "value": xs[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="__matmul__",
    x=_get_first_matrix_and_dtype(),
    y=_get_second_matrix_and_dtype(),
)
def test_numpy___matmul__(
    x,
    y,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    dtype1, x1 = x
    dtype2, x2 = y
    input_dtypes = dtype1 + dtype2

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": x1,
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "value": x2,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="__copy__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
    ),
)
def test_numpy___copy__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, x = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="__deepcopy__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
    ),
)
def test_numpy___deepcopy__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, x = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "memo": {},
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="__neg__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
    ),
)
def test_numpy___neg__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, x = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="__pos__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
    ),
)
def test_numpy___pos__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, x = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="__ifloordiv__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        allow_inf=False,
        large_abs_safety_factor=4,
        safety_factor_scale="linear",
        shared_dtype=True,
    ),
)
def test_numpy___ifloordiv__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, xs = dtype_and_x
    assume(not np.any(np.isclose(xs[1], 0)))
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": xs[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "value": xs[1],
        },
        init_flags=init_flags,
        method_flags=method_flags,
        frontend_method_data=frontend_method_data,
        frontend=frontend,
        atol_=1,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="__bool__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        max_dim_size=1,
    ),
)
def test_numpy___bool__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, x = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="__ne__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
)
def test_numpy___ne__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, xs = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": xs[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "value": xs[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="__eq__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
)
def test_numpy___eq__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, xs = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": xs[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "value": xs[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="__ge__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
)
def test_numpy___ge__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, xs = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": xs[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "value": xs[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="__gt__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
)
def test_numpy___gt__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, xs = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": xs[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "value": xs[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="__le__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
)
def test_numpy___le__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, xs = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": xs[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "value": xs[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="__lt__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
)
def test_numpy___lt__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, xs = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": xs[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "value": xs[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="__int__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_dim_size=1,
        max_dim_size=1,
    ),
)
def test_numpy___int__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, xs = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        method_input_dtypes=input_dtypes,
        init_all_as_kwargs_np={
            "object": xs[0],
        },
        method_all_as_kwargs_np={},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="__float__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_dim_size=1,
        max_dim_size=1,
    ),
)
def test_numpy___float__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, xs = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        method_input_dtypes=input_dtypes,
        init_all_as_kwargs_np={
            "object": xs[0],
        },
        method_all_as_kwargs_np={},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="__complex__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_dim_size=1,
        max_dim_size=1,
    ),
)
def test_numpy___complex__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, xs = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        method_input_dtypes=input_dtypes,
        init_all_as_kwargs_np={
            "object": xs[0],
        },
        method_all_as_kwargs_np={},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="__contains__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
)
def test_numpy___contains__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, xs = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        method_input_dtypes=input_dtypes,
        init_all_as_kwargs_np={
            "object": xs[0],
        },
        method_all_as_kwargs_np={
            "key": xs[0].reshape(-1)[0],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="__iadd__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"), num_arrays=2
    ),
)
def test_numpy___iadd__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, xs = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        method_input_dtypes=input_dtypes,
        init_all_as_kwargs_np={
            "object": xs[0],
        },
        method_all_as_kwargs_np={
            "value": xs[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="__isub__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"), num_arrays=2
    ),
)
def test_numpy___isub__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, xs = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        method_input_dtypes=input_dtypes,
        init_all_as_kwargs_np={
            "object": xs[0],
        },
        method_all_as_kwargs_np={
            "value": xs[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="__imul__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"), num_arrays=2
    ),
)
def test_numpy___imul__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, xs = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        method_input_dtypes=input_dtypes,
        init_all_as_kwargs_np={
            "object": xs[0],
        },
        method_all_as_kwargs_np={
            "value": xs[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# __itruediv__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="__itruediv__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_numpy___itruediv__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, xs = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": xs[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "value": xs[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="__ipow__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    power=helpers.ints(min_value=1, max_value=3),
)
def test_numpy___ipow__(
    dtype_and_x,
    power,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, xs = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        method_input_dtypes=input_dtypes,
        init_all_as_kwargs_np={
            "object": xs[0],
        },
        method_all_as_kwargs_np={
            "value": power,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="__iand__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("bool"),
        num_arrays=2,
    ),
)
def test_numpy___iand__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, xs = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        method_input_dtypes=input_dtypes,
        init_all_as_kwargs_np={
            "object": xs[0],
        },
        method_all_as_kwargs_np={
            "value": xs[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="__ior__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("bool"),
        num_arrays=2,
    ),
)
def test_numpy___ior__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, xs = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        method_input_dtypes=input_dtypes,
        init_all_as_kwargs_np={
            "object": xs[0],
        },
        method_all_as_kwargs_np={
            "value": xs[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="__ixor__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("bool"),
        num_arrays=2,
    ),
)
def test_numpy___ixor__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, xs = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        method_input_dtypes=input_dtypes,
        init_all_as_kwargs_np={
            "object": xs[0],
        },
        method_all_as_kwargs_np={
            "value": xs[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="__imod__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=0,
        exclude_min=True,
    ),
)
def test_numpy___imod__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, xs = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        method_input_dtypes=input_dtypes,
        init_all_as_kwargs_np={
            "object": xs[0],
        },
        method_all_as_kwargs_np={
            "value": xs[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="__abs__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        allow_inf=False,
        large_abs_safety_factor=4,
        safety_factor_scale="linear",
    ),
)
def test_numpy___abs__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, x = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# __len__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="__len__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=5,
    ),
)
def test_numpy___len__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={},
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        on_device=on_device,
    )


# __array__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="__array__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_numpy___array__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "dtype": input_dtypes[0],
        },
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        on_device=on_device,
    )


# __array_wrap__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="__array_wrap__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        num_arrays=2,
    ),
)
def test_numpy___array_wrap__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "array": x[1],
            "context": None,
        },
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="tobytes",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_numpy_tobytes(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={},
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        on_device=on_device,
    )


# tofile
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="tofile",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    path=st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Pc")),
        min_size=1,
        max_size=50,
    ),
)
def test_numpy_tofile(
    dtype_and_x,
    path,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "fid": path,
        },
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        on_device=on_device,
    )


# tolist
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="tolist",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_numpy_tolist(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={},
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        on_device=on_device,
        test_values=False,  # Todo change this after we add __iter__ to ndarray
    )


# __getitem__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="__getitem__",
    dtype_x_index=helpers.dtype_array_query(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_numpy_getitem(
    dtype_x_index,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtype, x, index = dtype_x_index
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtype[0]],
        init_all_as_kwargs_np={"object": x},
        method_input_dtypes=[*input_dtype[1:]],
        method_all_as_kwargs_np={"key": index},
        backend_to_test=backend_fw,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __setitem__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="__setitem__",
    dtypes_x_index_val=helpers.dtype_array_query_val(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_numpy_setitem(
    dtypes_x_index_val,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtype, x, index, val = dtypes_x_index_val
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtype[0]],
        init_all_as_kwargs_np={"object": x},
        method_input_dtypes=[*input_dtype[1:]],
        method_all_as_kwargs_np={"key": index, "value": val},
        backend_to_test=backend_fw,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# view
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="view",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_numpy_view(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={},
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        on_device=on_device,
    )


# mod
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="__mod__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=0,
        exclude_min=True,
    ),
)
def test_numpy___mod__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, xs = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        method_input_dtypes=input_dtypes,
        init_all_as_kwargs_np={
            "object": xs[0],
        },
        method_all_as_kwargs_np={
            "value": xs[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# ptp
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="ptp",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        valid_axis=True,
    ),
)
def test_numpy_ptp(
    dtype_x_axis,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, x, axis = dtype_x_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "axis": axis,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# item
@st.composite
def _item_helper(draw):
    dtype = draw(
        helpers.array_dtypes(
            num_arrays=1,
            available_dtypes=helpers.get_dtypes("numeric"),
        )
    )
    shape = draw(
        helpers.get_shape(
            min_num_dims=1,
            max_num_dims=3,
            min_dim_size=1,
            max_dim_size=10,
        )
    )
    array = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=shape,
            large_abs_safety_factor=2,
            small_abs_safety_factor=2,
        )
    )

    index = ()
    for s in shape:
        index += (draw(st.integers(min_value=-s + 1, max_value=s - 1)),)

    index_samples = [index, draw(helpers.ints(min_value=0, max_value=array.size - 1))]

    if array.size == 1:
        index_samples.append(None)

    sampled_index = draw(st.sampled_from(index_samples))

    if sampled_index is None:
        method_all_as_kwargs_np = {}
        num_positional_args = 0
    else:
        method_all_as_kwargs_np = {"args": sampled_index}
        num_positional_args = 1

    return dtype, array, method_all_as_kwargs_np, num_positional_args


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="item",
    args_kwargs=_item_helper(),
)
def test_numpy_instance_item(
    args_kwargs, frontend_method_data, init_flags, method_flags, frontend, on_device
):
    input_dtype, x, method_all_as_kwargs_np, num_positional_args = args_kwargs
    method_flags.num_positional_args = num_positional_args
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"object": x},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np=method_all_as_kwargs_np,
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="__rshift__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
    ),
)
def test_numpy___rshift__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtypes, x = dtype_and_x
    x[1] = np.asarray(
        np.clip(x[1], 0, np.iinfo(input_dtypes[1]).bits - 1), dtype=input_dtypes[1]
    )
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        method_all_as_kwargs_np={
            "value": x[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="__lshift__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        max_dim_size=1,
        max_value=2**31 - 1,
    ),
)
def test_numpy_instance_lshift__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtypes, x = dtype_and_x
    max_bits = np.iinfo(input_dtypes[0]).bits
    max_shift = max_bits - 1

    x[1] = np.asarray(np.clip(x[1], 0, max_shift), dtype=input_dtypes[1])

    max_value_before_shift = 2 ** (max_bits - x[1]) - 1
    overflow_threshold = 2 ** (max_bits - 1)

    x[0] = np.asarray(
        np.clip(x[0], None, max_value_before_shift), dtype=input_dtypes[0]
    )

    if np.any(x[0] > overflow_threshold):
        x[0] = np.clip(x[0], None, overflow_threshold)
    if np.any(x[0] < 0):
        x[0] = np.abs(x[0])

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "value": x[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# __tostring__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="tostring",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_numpy_tostring(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtypes, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={},
        backend_to_test=backend_fw,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="__invert__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes(kind="integer"),
        num_arrays=1,
    ),
)
def test_numpy___invert__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtypes, xs = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        init_all_as_kwargs_np={
            "object": xs[0],
        },
        method_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        method_all_as_kwargs_np={},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# trace
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.array",
    method_name="trace",
    dtype_and_x_axes=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        valid_axis=True,
        min_axes_size=2,
        max_axes_size=2,
        min_num_dims=2,
        large_abs_safety_factor=24,
        small_abs_safety_factor=24,
        safety_factor_scale="log",
    ),
    offset=st.integers(min_value=-4, max_value=4),
)
def test_numpy_ndarray_trace(
    dtype_and_x_axes,
    offset,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtypes, x, axes = dtype_and_x_axes

    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "offset": offset,
            "axis1": axes[0],
            "axis2": axes[1],
        },
        backend_to_test=backend_fw,
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )
