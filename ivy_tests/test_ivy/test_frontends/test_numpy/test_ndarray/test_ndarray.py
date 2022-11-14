# global
import pytest
import numpy as np
from hypothesis import strategies as st

import ivy
# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_method, assert_all_close
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.test_functional.test_core.test_linalg import (
    _get_first_matrix_and_dtype,
    _get_second_matrix_and_dtype,
)


pytestmark = pytest.mark.skip("handle_frontend_method decorator wip")


@handle_frontend_method(
    method_tree="numpy.ndarray.argmax",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
    ),
)
def test_numpy_ndarray_argmax(
    dtype_and_x,
    as_variable,
    native_array,
    class_,
    method_name,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_as_variable_flags=as_variable,
        init_num_positional_args=0,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[input_dtype[1]],
        method_as_variable_flags=as_variable,
        method_num_positional_args=0,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "value": x[1],
        },
        frontend="numpy",
        class_="ndarray",
        method_name="argmax",
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
    method_tree="numpy.ndarray.reshape",
    dtypes_x_shape=dtypes_x_reshape(),
)
def test_numpy_ndarray_reshape(
    dtypes_x_shape,
    as_variable,
    native_array,
    class_,
    method_name,
):
    input_dtype, x, shape = dtypes_x_shape
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=0,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_as_variable_flags=as_variable,
        method_num_positional_args=0,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "shape": shape,
        },
        frontend="numpy",
        class_="ndarray",
        method_name="reshape",
    )


@handle_frontend_method(
    method_tree="numpy.ndarray.transpose",
    array_and_axes=np_frontend_helpers._array_and_axes_permute_helper(
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=2,
        max_dim_size=10,
    ),
)
def test_numpy_ndarray_transpose(
    array_and_axes,
    as_variable,
    num_positional_args,
    native_array,
    class_,
    method_name,
):
    array, dtype, axes = array_and_axes
    helpers.test_frontend_method(
        init_input_dtypes=dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": np.array(array),
        },
        method_input_dtypes=dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "axes": axes,
        },
        frontend="numpy",
        class_="ndarray",
        method_name="transpose",
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
            sorted=False,
            unique=True,
            min_size=2,
            max_size=2,
            force_tuple=True,
        )
    )
    return dtype, x, axis1, axis2


@handle_frontend_method(
    method_tree="numpy.ndarray.swapaxes",
    dtype_x_and_axes=dtype_values_and_axes(),
)
def test_numpy_ndarray_swapaxes(
    dtype_x_and_axes,
    as_variable,
    native_array,
    num_positional_args_method,
    frontend,
    class_,
    method_name,
):
    input_dtype, x, axis1, axis2 = dtype_x_and_axes
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        input_dtypes_method=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        num_positional_args_method=num_positional_args_method,
        native_array_flags_init=native_array,
        as_variable_flags_method=as_variable,
        native_array_flags_method=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        all_as_kwargs_np_method={
            "axis1": axis1,
            "axis2": axis2,
        },
        frontend=frontend,
        class_=class_,
        method_name=method_name,
    )


# any
@handle_frontend_method(
    method_tree="numpy.ndarray.any",
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
def test_numpy_ndarray_any(
    dtype_x_axis,
    keepdims,
    where,
    as_variable,
    num_positional_args,
    native_array,
    class_,
    method_name,
):
    input_dtype, x, axis = dtype_x_axis
    where, as_variable, native_array = np_frontend_helpers.handle_where_and_array_bools(
        where=[where[0][0]] if isinstance(where, list) else where,
        input_dtype=input_dtype,
        as_variable=as_variable,
        native_array=native_array,
    )
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "axis": axis,
            "out": None,
            "keepdims": keepdims,
            "where": where,
        },
        frontend="numpy",
        class_="ndarray",
        method_name="any",
    )


@handle_frontend_method(
    method_tree="numpy.ndarray.all",
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
    as_variable=helpers.array_bools(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.ndarray.all"
    ),
    native_array=helpers.array_bools(),
)
def test_numpy_ndarray_all(
    dtype_x_axis,
    keepdims,
    where,
    as_variable,
    num_positional_args,
    native_array,
    class_,
    method_name,
):
    input_dtype, x, axis = dtype_x_axis
    where, as_variable, native_array = np_frontend_helpers.handle_where_and_array_bools(
        where=[where[0][0]] if isinstance(where, list) else where,
        input_dtype=input_dtype,
        as_variable=as_variable,
        native_array=native_array,
    )
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "axis": axis,
            "out": None,
            "keepdims": keepdims,
            "where": where,
        },
        frontend="numpy",
        class_="ndarray",
        method_name="all",
    )


@handle_frontend_method(
    method_tree="numpy.ndarray.argsort",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
)
def test_numpy_instance_argsort(
    dtype_x_axis,
    as_variable,
    num_positional_args,
    native_array,
    fw,
    class_,
    method_name,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        frontend_class=np.ndarray,
        fn_tree="ndarray.argsort",
        x=x[0],
        axis=axis,
    )


@handle_frontend_method(
    method_tree="numpy.ndarray.mean",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
)
def test_numpy_ndarray_mean(
    dtype_x_axis,
    as_variable,
    num_positional_args_method,
    native_array,
    class_,
    method_name,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        input_dtypes_method=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        num_positional_args_method=num_positional_args_method,
        native_array_flags_init=native_array,
        as_variable_flags_method=as_variable,
        native_array_flags_method=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        all_as_kwargs_np_method={
            "axis": axis,
            "dtype": "float64",
            "out": None,
        },
        frontend="numpy",
        class_="ndarray",
        method_name="mean",
    )


@handle_frontend_method(
    method_tree="numpy.ndarray.min",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
    keepdims=st.booleans(),
)
def test_numpy_instance_min(
    dtype_x_axis,
    keepdims,
    as_variable,
    num_positional_args_method,
    native_array,
    class_,
    method_name,
):
    input_dtype, x, axis = dtype_x_axis

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=num_positional_args_method,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "axis": axis,
            "keepdims": keepdims,
        },
        frontend="numpy",
        class_="ndarray",
        method_name="min",
    )


@handle_frontend_method(
    method_tree="numpy.ndarray.argmin",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
    ),
    keepdims=st.booleans(),
)
def test_numpy_ndarray_argmin(
    dtype_x_axis,
    keepdims,
    as_variable,
    num_positional_args,
    native_array,
    fw,
    class_,
    method_name,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        all_as_kwargs_np_method={
            "axis": axis,
            "keepdims": keepdims,
        },
        fw=fw,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        frontend="numpy",
        class_="ndarray",
        method_name="argmin",
        frontend_class=np.ndarray,
        fn_tree="ndarray.argmin",
    )


@handle_frontend_method(
    method_tree="numpy.ndarray.clip",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=2,
    ),
)
def test_numpy_instance_clip(
    dtype_and_x,
    as_variable,
    num_positional_args_method,
    native_array,
    class_,
    method_name,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        input_dtypes_method=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        num_positional_args_method=num_positional_args_method,
        native_array_flags_init=native_array,
        as_variable_flags_method=as_variable,
        native_array_flags_method=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        all_as_kwargs_np_method={"a_min": 0, "a_max": 1},
        frontend="numpy",
        class_="ndarray",
        method_name="clip",
    )


@handle_frontend_method(
    method_tree="numpy.ndarray.max",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
    keepdims=st.booleans(),
)
def test_numpy_instance_max(
    dtype_x_axis,
    keepdims,
    as_variable,
    num_positional_args_method,
    native_array,
    class_,
    method_name,
):
    input_dtype, x, axis = dtype_x_axis

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=num_positional_args_method,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "axis": axis,
            "keepdims": keepdims,
        },
        frontend="numpy",
        class_="ndarray",
        method_name="max",
    )


@handle_frontend_method(
    method_tree="numpy.ndarray.cumprod",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
    dtype=helpers.get_dtypes("float", full=False, none=True),
)
def test_numpy_instance_cumprod(
    dtype_x_axis,
    dtype,
    as_variable,
    num_positional_args_method,
    native_array,
    class_,
    method_name,
):
    input_dtype, x, axis = dtype_x_axis

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=num_positional_args_method,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "axis": axis,
            "dtype": dtype[0],
            "out": None,
        },
        frontend="numpy",
        class_="ndarray",
        method_name="cumprod",
    )


@handle_frontend_method(
    method_tree="numpy.ndarray.cumsum",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
    dtype=helpers.get_dtypes("float", full=False, none=True),
)
def test_numpy_instance_cumsum(
    dtype_x_axis,
    dtype,
    as_variable,
    num_positional_args_method,
    native_array,
    class_,
    method_name,
):
    input_dtype, x, axis = dtype_x_axis

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=num_positional_args_method,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "axis": axis,
            "dtype": dtype[0],
            "out": None,
        },
        frontend="numpy",
        class_="ndarray",
        method_name="cumsum",
    )


@handle_frontend_method(
    method_tree="numpy.ndarray.sort",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
)
def test_numpy_instance_sort(
    dtype_x_axis,
    as_variable,
    num_positional_args_method,
    native_array,
    class_,
    method_name,
):
    input_dtype, x, axis = dtype_x_axis

    ret, frontend_ret = helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=num_positional_args_method,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "axis": axis,
        },
        frontend="numpy",
        class_="ndarray",
        method_name="sort",
        test_values=False,
    )
    frontend_ret = np.sort(x[0], axis=axis)
    assert_all_close(
        ret_np=ret,
        ret_from_gt_np=frontend_ret,
        rtol=1e-2,
        atol=1e-2,
        ground_truth_backend="numpy",
    )


@handle_frontend_method(
    method_tree="numpy.ndarray.copy",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
    ),
)
def test_numpy_instance_copy(
    dtype_and_x,
    as_variable,
    num_positional_args_method,
    native_array,
    class_,
    method_name,
):
    input_dtype, x = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=num_positional_args_method,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend="numpy",
        class_="ndarray",
        method_name="copy",
    )


@handle_frontend_method(
    method_tree="numpy.ndarray.nonzero",
    dtype_and_a=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_numpy_instance_nonzero(
    dtype_and_a,
    as_variable,
    num_positional_args_method,
    native_array,
    class_,
    method_name,
):
    input_dtype, a = dtype_and_a

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": a[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=num_positional_args_method,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend="numpy",
        class_="ndarray",
        method_name="nonzero",
    )


@handle_frontend_method(
    method_tree="numpy.ndarray.ravel",
    dtype_and_a=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_numpy_instance_ravel(
    dtype_and_a,
    as_variable,
    num_positional_args_method,
    native_array,
    class_,
    method_name,
):
    input_dtype, a = dtype_and_a

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": a[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=num_positional_args_method,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend="numpy",
        class_="ndarray",
        method_name="ravel",
    )


@handle_frontend_method(
    method_tree="numpy.ndarray.repeat",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=2,
        min_dim_size=2,
    ),
    repeats=helpers.ints(min_value=2, max_value=5),
    axis=helpers.ints(min_value=-1, max_value=1),
)
def test_numpy_instance_repeat(
    dtype_and_x,
    repeats,
    axis,
    as_variable,
    num_positional_args_method,
    native_array,
    class_,
    method_name,
):
    input_dtype, x = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=num_positional_args_method,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "repeats": repeats,
            "axis": axis,
        },
        frontend="numpy",
        class_="ndarray",
        method_name="repeat",
    )


@handle_frontend_method(
    method_tree="numpy.ndarray.searchsorted",
    dtype_x_v=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("signed_integer"),
        min_num_dims=1,
        max_num_dims=1,
        num_arrays=2,
    ),
    side=st.sampled_from(["left", "right"]),
)
def test_numpy_instance_searchsorted(
    dtype_x_v,
    side,
    as_variable,
    num_positional_args_method,
    native_array,
    class_,
    method_name,
):
    input_dtype, xs = dtype_x_v

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": xs[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=num_positional_args_method,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "v": xs[1],
            "side": side,
            "sorter": np.argsort(xs[0]),
        },
        frontend="numpy",
        class_="ndarray",
        method_name="searchsorted",
    )


@handle_frontend_method(
    method_tree="numpy.ndarray.squeeze",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
)
def test_numpy_instance_squeeze(
    dtype_x_axis,
    as_variable,
    num_positional_args_method,
    native_array,
    class_,
    method_name,
):
    input_dtype, x, axis = dtype_x_axis

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=num_positional_args_method,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "axis": axis,
        },
        frontend="numpy",
        class_="ndarray",
        method_name="squeeze",
    )

@handle_cmd_line_args
@given(
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
    num_positional_args_method=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.ndarray.std"
    )
)
def test_numpy_instance_std(
    dtype_x_axis,
    keepdims,
    where,
    as_variable,
    num_positional_args_method,
    native_array,
):
    input_dtype, x, axis = dtype_x_axis
    where, as_variable, native_array = np_frontend_helpers.handle_where_and_array_bools(
        where=[where[0][0]] if isinstance(where, list) else where,
        input_dtype=input_dtype,
        as_variable=as_variable,
        native_array=native_array,
    )
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        input_dtypes_method=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=num_positional_args_method,
        num_positional_args_method=num_positional_args_method,
        native_array_flags_init=native_array,
        as_variable_flags_method=as_variable,
        native_array_flags_method=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        all_as_kwargs_np_method={
            "axis": axis,
            "out": None,
            "ddof" : 0,
            "keepdims": keepdims,
            "where": where,
        },
        frontend="numpy",
        class_name="ndarray",
        method_name="std",
    )



@handle_frontend_method(
    method_tree="numpy.ndarray.__add__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"), num_arrays=2
    ),
    num_positional_args_method=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.ndarray.__add__"
    ),
)
def test_numpy_instance_add__(
    dtype_and_x,
    as_variable,
    num_positional_args_method,
    native_array,
    class_,
    method_name,
):
    input_dtype, xs = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": xs[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=num_positional_args_method,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "value": xs[1],
        },
        frontend="numpy",
        class_="ndarray",
        method_name="__add__",
    )


@handle_frontend_method(
    method_tree="numpy.ndarray.__sub__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"), num_arrays=2
    ),
)
def test_numpy_instance_sub__(
    dtype_and_x,
    as_variable,
    num_positional_args_method,
    native_array,
    class_,
    method_name,
):
    input_dtype, xs = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": xs[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=num_positional_args_method,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "value": xs[1],
        },
        frontend="numpy",
        class_="ndarray",
        method_name="__sub__",
    )


@handle_frontend_method(
    method_tree="numpy.ndarray.__mul__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
)
def test_numpy_instance_mul__(
    dtype_and_x,
    as_variable,
    num_positional_args_method,
    native_array,
    class_,
    method_name,
):
    input_dtype, xs = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": xs[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=num_positional_args_method,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "value": xs[1],
        },
        frontend="numpy",
        class_="ndarray",
        method_name="__mul__",
    )


@handle_frontend_method(
    method_tree="numpy.ndarray.__and__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("bool"),
        num_arrays=2,
    ),
)
def test_numpy_instance_and__(
    dtype_and_x,
    as_variable,
    num_positional_args_method,
    native_array,
    class_,
    method_name,
):
    input_dtype, xs = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": xs[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=num_positional_args_method,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "value": xs[1],
        },
        frontend="numpy",
        class_="ndarray",
        method_name="__and__",
    )


@handle_frontend_method(
    method_tree="numpy.ndarray.__or__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("bool"),
        num_arrays=2,
    ),
)
def test_numpy_instance_or__(
    dtype_and_x,
    as_variable,
    num_positional_args_method,
    native_array,
    class_,
    method_name,
):
    input_dtype, xs = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": xs[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=num_positional_args_method,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "value": xs[1],
        },
        frontend="numpy",
        class_="ndarray",
        method_name="__or__",
    )


@handle_frontend_method(
    method_tree="numpy.ndarray.__xor__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("bool"),
        num_arrays=2,
    ),
)
def test_numpy_instance_xor__(
    dtype_and_x,
    as_variable,
    num_positional_args_method,
    native_array,
    class_,
    method_name,
):
    input_dtype, xs = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": xs[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=num_positional_args_method,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "value": xs[1],
        },
        frontend="numpy",
        class_="ndarray",
        method_name="__xor__",
    )


@handle_frontend_method(
    method_tree="numpy.ndarray.__matmul__",
    x=_get_first_matrix_and_dtype(),
    y=_get_second_matrix_and_dtype(),
)
def test_numpy_instance_matmul__(
    x,
    y,
    as_variable,
    num_positional_args_method,
    native_array,
    class_,
    method_name,
):
    dtype1, x1 = x
    dtype2, x2 = y
    input_dtype = dtype1 + dtype2

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x1,
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=num_positional_args_method,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "value": x2,
        },
        frontend="numpy",
        class_="ndarray",
        method_name="__matmul__",
    )


@handle_frontend_method(
    method_tree="numpy.ndarray.__copy__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
    ),
)
def test_numpy_instance_copy__(
    dtype_and_x,
    as_variable,
    num_positional_args_method,
    native_array,
    class_,
    method_name,
):
    input_dtype, x = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=num_positional_args_method,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend="numpy",
        class_="ndarray",
        method_name="__copy__",
    )


@handle_frontend_method(
    method_tree="numpy.ndarray.__neg__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
    ),
)
def test_numpy_instance_neg__(
    dtype_and_x,
    as_variable,
    num_positional_args_method,
    native_array,
    class_,
    method_name,
):
    input_dtype, x = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=num_positional_args_method,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend="numpy",
        class_="ndarray",
        method_name="__neg__",
    )


@handle_frontend_method(
    method_tree="numpy.ndarray.__pos__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
    ),
)
def test_numpy_instance_pos__(
    dtype_and_x,
    as_variable,
    num_positional_args_method,
    native_array,
    class_,
    method_name,
):
    input_dtype, x = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=num_positional_args_method,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend="numpy",
        class_="ndarray",
        method_name="__pos__",
    )


@handle_frontend_method(
    method_tree="numpy.ndarray.__bool__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        max_dim_size=1,
    ),
)
def test_numpy_instance_bool__(
    dtype_and_x,
    as_variable,
    num_positional_args_method,
    native_array,
    class_,
    method_name,
):
    input_dtype, x = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=num_positional_args_method,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend="numpy",
        class_="ndarray",
        method_name="__bool__",
    )


@handle_frontend_method(
    method_tree="numpy.ndarray.__ne__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
)
def test_numpy_instance_ne__(
    dtype_and_x,
    as_variable,
    num_positional_args_method,
    native_array,
    class_,
    method_name,
):
    input_dtype, xs = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": xs[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=num_positional_args_method,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "value": xs[1],
        },
        frontend="numpy",
        class_="ndarray",
        method_name="__ne__",
    )


@handle_frontend_method(
    method_tree="numpy.ndarray.__eq__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
)
def test_numpy_instance_eq__(
    dtype_and_x,
    as_variable,
    num_positional_args_method,
    native_array,
    class_,
    method_name,
):
    input_dtype, xs = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": xs[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=num_positional_args_method,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "value": xs[1],
        },
        frontend="numpy",
        class_="ndarray",
        method_name="__eq__",
    )


@handle_frontend_method(
    method_tree="numpy.ndarray.__ge__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
)
def test_numpy_instance_ge__(
    dtype_and_x,
    as_variable,
    num_positional_args_method,
    native_array,
    class_,
    method_name,
):
    input_dtype, xs = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": xs[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=num_positional_args_method,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "value": xs[1],
        },
        frontend="numpy",
        class_="ndarray",
        method_name="__ge__",
    )


@handle_frontend_method(
    method_tree="numpy.ndarray.__gt__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
)
def test_numpy_instance_gt__(
    dtype_and_x,
    as_variable,
    num_positional_args_method,
    native_array,
    class_,
    method_name,
):
    input_dtype, xs = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": xs[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=num_positional_args_method,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "value": xs[1],
        },
        frontend="numpy",
        class_="ndarray",
        method_name="__gt__",
    )


@handle_frontend_method(
    method_tree="numpy.ndarray.__le__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
)
def test_numpy_instance_le__(
    dtype_and_x,
    as_variable,
    num_positional_args_method,
    native_array,
    class_,
    method_name,
):
    input_dtype, xs = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": xs[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=num_positional_args_method,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "value": xs[1],
        },
        frontend="numpy",
        class_="ndarray",
        method_name="__le__",
    )


@handle_frontend_method(
    method_tree="numpy.ndarray.__lt__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
)
def test_numpy_instance_lt__(
    dtype_and_x,
    as_variable,
    num_positional_args_method,
    native_array,
    class_,
    method_name,
):
    input_dtype, xs = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": xs[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=num_positional_args_method,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "value": xs[1],
        },
        frontend="numpy",
        class_="ndarray",
        method_name="__lt__",
    )


@handle_frontend_method(
    method_tree="numpy.ndarray.__int__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_dim_size=1,
        max_dim_size=1,
    ),
)
def test_numpy_instance_int__(
    dtype_and_x,
    as_variable,
    num_positional_args_method,
    native_array,
    class_,
    frontend,
    method_name,
):
    input_dtype, xs = dtype_and_x

    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        input_dtypes_method=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        num_positional_args_method=num_positional_args_method,
        native_array_flags_init=native_array,
        as_variable_flags_method=as_variable,
        native_array_flags_method=native_array,
        all_as_kwargs_np_init={
            "data": xs[0],
        },
        all_as_kwargs_np_method={},
        frontend=frontend,
        class_=class_,
        method_name=method_name,
    )


@handle_frontend_method(
    method_tree="numpy.ndarray.__float__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_dim_size=1,
        max_dim_size=1,
    ),
)
def test_numpy_instance_float__(
    dtype_and_x,
    as_variable,
    num_positional_args_method,
    native_array,
    class_,
    frontend,
    method_name,
):
    input_dtype, xs = dtype_and_x

    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        input_dtypes_method=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        num_positional_args_method=num_positional_args_method,
        native_array_flags_init=native_array,
        as_variable_flags_method=as_variable,
        native_array_flags_method=native_array,
        all_as_kwargs_np_init={
            "data": xs[0],
        },
        all_as_kwargs_np_method={},
        frontend=frontend,
        class_=class_,
        method_name=method_name,
    )


@handle_frontend_method(
    method_tree="numpy.ndarray.__contains__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
)
def test_numpy_instance_contains__(
    dtype_and_x,
    as_variable,
    num_positional_args_method,
    native_array,
    class_,
    frontend,
    method_name,
):
    input_dtype, xs = dtype_and_x

    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        input_dtypes_method=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        num_positional_args_method=num_positional_args_method,
        native_array_flags_init=native_array,
        as_variable_flags_method=as_variable,
        native_array_flags_method=native_array,
        all_as_kwargs_np_init={
            "data": xs[0],
        },
        all_as_kwargs_np_method={
            "key": xs[0].reshape(-1)[0],
        },
        frontend=frontend,
        class_=class_,
        method_name=method_name,
    )


@handle_frontend_method(
    method_tree="numpy.ndarray.__iadd__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"), num_arrays=2
    ),
)
def test_numpy_instance_iadd__(
    dtype_and_x,
    as_variable,
    num_positional_args_method,
    native_array,
    class_,
    frontend,
    method_name,
):
    input_dtype, xs = dtype_and_x

    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        input_dtypes_method=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        num_positional_args_method=num_positional_args_method,
        native_array_flags_init=native_array,
        as_variable_flags_method=as_variable,
        native_array_flags_method=native_array,
        all_as_kwargs_np_init={
            "data": xs[0],
        },
        all_as_kwargs_np_method={
            "value": xs[1],
        },
        frontend=frontend,
        class_=class_,
        method_name=method_name,
    )


@handle_frontend_method(
    method_tree="numpy.ndarray.__isub__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"), num_arrays=2
    ),
)
def test_numpy_instance_isub__(
    dtype_and_x,
    as_variable,
    num_positional_args_method,
    native_array,
    class_,
    frontend,
    method_name,
):
    input_dtype, xs = dtype_and_x

    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        input_dtypes_method=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        num_positional_args_method=num_positional_args_method,
        native_array_flags_init=native_array,
        as_variable_flags_method=as_variable,
        native_array_flags_method=native_array,
        all_as_kwargs_np_init={
            "data": xs[0],
        },
        all_as_kwargs_np_method={
            "value": xs[1],
        },
        frontend=frontend,
        class_=class_,
        method_name=method_name,
    )


@handle_frontend_method(
    method_tree="numpy.ndarray.__imul__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"), num_arrays=2
    ),
)
def test_numpy_instance_imul__(
    dtype_and_x,
    as_variable,
    num_positional_args_method,
    native_array,
    class_,
    frontend,
    method_name,
):
    input_dtype, xs = dtype_and_x

    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        input_dtypes_method=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        num_positional_args_method=num_positional_args_method,
        native_array_flags_init=native_array,
        as_variable_flags_method=as_variable,
        native_array_flags_method=native_array,
        all_as_kwargs_np_init={
            "data": xs[0],
        },
        all_as_kwargs_np_method={
            "value": xs[1],
        },
        frontend=frontend,
        class_=class_,
        method_name=method_name,
    )


@handle_frontend_method(
    method_tree="numpy.ndarray.__ipow__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    power=helpers.ints(min_value=1, max_value=3),
)
def test_numpy_instance_ipow__(
    dtype_and_x,
    power,
    as_variable,
    num_positional_args_method,
    native_array,
    class_,
    frontend,
    method_name,
):
    input_dtype, xs = dtype_and_x

    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        input_dtypes_method=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        num_positional_args_method=num_positional_args_method,
        native_array_flags_init=native_array,
        as_variable_flags_method=as_variable,
        native_array_flags_method=native_array,
        all_as_kwargs_np_init={
            "data": xs[0],
        },
        all_as_kwargs_np_method={
            "value": power,
        },
        frontend=frontend,
        class_=class_,
        method_name=method_name,
    )


@handle_frontend_method(
    method_tree="numpy.ndarray.__iand__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("bool"),
        num_arrays=2,
    ),
)
def test_numpy_instance_iand__(
    dtype_and_x,
    as_variable,
    num_positional_args_method,
    native_array,
    class_,
    frontend,
    method_name,
):
    input_dtype, xs = dtype_and_x

    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        input_dtypes_method=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        num_positional_args_method=num_positional_args_method,
        native_array_flags_init=native_array,
        as_variable_flags_method=as_variable,
        native_array_flags_method=native_array,
        all_as_kwargs_np_init={
            "data": xs[0],
        },
        all_as_kwargs_np_method={
            "value": xs[1],
        },
        frontend=frontend,
        class_=class_,
        method_name=method_name,
    )


@handle_frontend_method(
    method_tree="numpy.ndarray.__ior__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("bool"),
        num_arrays=2,
    ),
)
def test_numpy_instance_ior__(
    dtype_and_x,
    as_variable,
    num_positional_args_method,
    native_array,
    class_,
    frontend,
    method_name,
):
    input_dtype, xs = dtype_and_x

    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        input_dtypes_method=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        num_positional_args_method=num_positional_args_method,
        native_array_flags_init=native_array,
        as_variable_flags_method=as_variable,
        native_array_flags_method=native_array,
        all_as_kwargs_np_init={
            "data": xs[0],
        },
        all_as_kwargs_np_method={
            "value": xs[1],
        },
        frontend=frontend,
        class_=class_,
        method_name=method_name,
    )


@handle_frontend_method(
    method_tree="numpy.ndarray.__ixor__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("bool"),
        num_arrays=2,
    ),
)
def test_numpy_instance_ixor__(
    dtype_and_x,
    as_variable,
    num_positional_args_method,
    native_array,
    class_,
    frontend,
    method_name,
):
    input_dtype, xs = dtype_and_x

    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        input_dtypes_method=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        num_positional_args_method=num_positional_args_method,
        native_array_flags_init=native_array,
        as_variable_flags_method=as_variable,
        native_array_flags_method=native_array,
        all_as_kwargs_np_init={
            "data": xs[0],
        },
        all_as_kwargs_np_method={
            "value": xs[1],
        },
        frontend=frontend,
        class_=class_,
        method_name=method_name,
    )


@handle_frontend_method(
    method_tree="numpy.ndarray.__imod__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=0,
        exclude_min=True,
    ),
)
def test_numpy_instance_imod__(
    dtype_and_x,
    as_variable,
    num_positional_args_method,
    native_array,
    class_,
    frontend,
    method_name,
):
    input_dtype, xs = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        input_dtypes_method=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        num_positional_args_method=num_positional_args_method,
        native_array_flags_init=native_array,
        as_variable_flags_method=as_variable,
        native_array_flags_method=native_array,
        all_as_kwargs_np_init={
            "data": xs[0],
        },
        all_as_kwargs_np_method={
            "value": xs[1],
        },
        frontend=frontend,
        class_=class_,
        method_name=method_name,
    )
