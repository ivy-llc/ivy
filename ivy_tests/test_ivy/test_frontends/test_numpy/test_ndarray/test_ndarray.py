# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args, assert_all_close
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.test_functional.test_core.test_linalg import (
    _get_first_matrix_and_dtype,
    _get_second_matrix_and_dtype,
)


# argmax
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
    ),
)
def test_numpy_ndarray_argmax(
    dtype_and_x,
    as_variable,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtype_init=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=0,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=[input_dtype[1]],
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={
            "value": x[1],
        },
        fw=fw,
        frontend="numpy",
        class_name="ndarray",
        method_name="argmax",
    )


# reshape
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


@handle_cmd_line_args
@given(
    dtypes_x_shape=dtypes_x_reshape(),
)
def test_numpy_ndarray_reshape(
    dtypes_x_shape,
    as_variable,
    native_array,
):
    input_dtype, x, shape = dtypes_x_shape
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=0,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=[],
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={
            "shape": shape,
        },
        frontend="numpy",
        class_name="ndarray",
        method_name="reshape",
    )


# transpose
@handle_cmd_line_args
@given(
    array_and_axes=np_frontend_helpers._array_and_axes_permute_helper(
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=2,
        max_dim_size=10,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.ndarray.transpose"
    ),
)
def test_numpy_ndarray_transpose(
    array_and_axes,
    as_variable,
    num_positional_args,
    native_array,
):
    array, dtype, axes = array_and_axes
    helpers.test_frontend_method(
        input_dtypes_init=dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=num_positional_args,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": np.array(array),
        },
        input_dtypes_method=dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=num_positional_args,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={
            "axes": axes,
        },
        frontend="numpy",
        class_name="ndarray",
        method_name="transpose",
    )


# any
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
    as_variable=helpers.array_bools(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.ndarray.any"
    ),
    native_array=helpers.array_bools(),
)
def test_numpy_ndarray_any(
    dtype_x_axis,
    keepdims,
    where,
    as_variable,
    num_positional_args,
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
        as_variable_flags_init=as_variable,
        num_positional_args_init=num_positional_args,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=num_positional_args,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={
            "axis": axis,
            "out": None,
            "keepdims": keepdims,
            "where": where,
        },
        frontend="numpy",
        class_name="ndarray",
        method_name="any",
    )


# all
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
        num_positional_args_init=num_positional_args,
        num_positional_args_method=num_positional_args,
        native_array_flags_init=native_array,
        as_variable_flags_method=as_variable,
        native_array_flags_method=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        all_as_kwargs_np_method={
            "axis": axis,
            "out": None,
            "keepdims": keepdims,
            "where": where,
        },
        frontend="numpy",
        class_name="ndarray",
        method_name="all",
    )


@handle_cmd_line_args
@given(
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.ndarray.argsort"
    ),
)
def test_numpy_instance_argsort(
    dtype_x_axis,
    as_variable,
    num_positional_args,
    native_array,
    fw,
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


@handle_cmd_line_args
@given(
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
    keepdims=st.booleans(),
    num_positional_args_method=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.ndarray.min"
    ),
)
def test_numpy_instance_min(
    dtype_x_axis,
    keepdims,
    as_variable,
    num_positional_args_method,
    native_array,
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
            "keepdims": keepdims,
        },
        frontend="numpy",
        class_name="ndarray",
        method_name="min",
    )


# argmin
@handle_cmd_line_args
@given(
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
    ),
    keepdims=st.booleans(),
    num_positional_args_method=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.ndarray.argmin"
    ),
)
def test_numpy_ndarray_argmin(
    dtype_x_axis,
    keepdims,
    as_variable,
    num_positional_args,
    native_array,
    fw,
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
        class_name="ndarray",
        method_name="argmin",
        frontend_class=np.ndarray,
        fn_tree="ndarray.argmin",
    )


@handle_cmd_line_args
@given(
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
    keepdims=st.booleans(),
    num_positional_args_method=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.ndarray.max"
    ),
)
def test_numpy_instance_max(
    dtype_x_axis,
    keepdims,
    as_variable,
    num_positional_args_method,
    native_array,
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
            "keepdims": keepdims,
        },
        frontend="numpy",
        class_name="ndarray",
        method_name="max",
    )


@handle_cmd_line_args
@given(
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
    dtype=helpers.get_dtypes("float", full=False, none=True),
    num_positional_args_method=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.ndarray.cumprod"
    ),
)
def test_numpy_instance_cumprod(
    dtype_x_axis,
    dtype,
    as_variable,
    num_positional_args_method,
    native_array,
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
            "dtype": dtype[0],
            "out": None,
        },
        frontend="numpy",
        class_name="ndarray",
        method_name="cumprod",
    )


@handle_cmd_line_args
@given(
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
    dtype=helpers.get_dtypes("float", full=False, none=True),
    num_positional_args_method=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.ndarray.cumsum"
    ),
)
def test_numpy_instance_cumsum(
    dtype_x_axis,
    dtype,
    as_variable,
    num_positional_args_method,
    native_array,
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
            "dtype": dtype[0],
            "out": None,
        },
        frontend="numpy",
        class_name="ndarray",
        method_name="cumsum",
    )


@handle_cmd_line_args
@given(
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
    num_positional_args_method=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.ndarray.sort"
    ),
)
def test_numpy_instance_sort(
    dtype_x_axis,
    as_variable,
    num_positional_args_method,
    native_array,
):
    input_dtype, x, axis = dtype_x_axis

    ret, frontend_ret = helpers.test_frontend_method(
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
        },
        frontend="numpy",
        class_name="ndarray",
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


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
    ),
    num_positional_args_method=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.ndarray.copy"
    ),
)
def test_numpy_instance_copy(
    dtype_and_x,
    as_variable,
    num_positional_args_method,
    native_array,
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
        all_as_kwargs_np_method={},
        frontend="numpy",
        class_name="ndarray",
        method_name="copy",
    )


@handle_cmd_line_args
@given(
    dtype_and_a=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    num_positional_args_method=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.ndarray.nonzero"
    ),
)
def test_numpy_instance_nonzero(
    dtype_and_a,
    as_variable,
    num_positional_args_method,
    native_array,
):
    input_dtype, a = dtype_and_a

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
            "data": a[0],
        },
        all_as_kwargs_np_method={},
        frontend="numpy",
        class_name="ndarray",
        method_name="nonzero",
    )


@handle_cmd_line_args
@given(
    dtype_and_a=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    num_positional_args_method=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.ndarray.ravel"
    ),
)
def test_numpy_instance_ravel(
    dtype_and_a,
    as_variable,
    num_positional_args_method,
    native_array,
):
    input_dtype, a = dtype_and_a

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
            "data": a[0],
        },
        all_as_kwargs_np_method={},
        frontend="numpy",
        class_name="ndarray",
        method_name="ravel",
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=2,
        min_dim_size=2,
    ),
    repeats=helpers.ints(min_value=2, max_value=5),
    axis=helpers.ints(min_value=-1, max_value=1),
    num_positional_args_method=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.ndarray.repeat"
    ),
)
def test_numpy_instance_repeat(
    dtype_and_x,
    repeats,
    axis,
    as_variable,
    num_positional_args_method,
    native_array,
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
        all_as_kwargs_np_method={
            "repeats": repeats,
            "axis": axis,
        },
        frontend="numpy",
        class_name="ndarray",
        method_name="repeat",
    )


@handle_cmd_line_args
@given(
    dtype_x_v=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("signed_integer"),
        min_num_dims=1,
        max_num_dims=1,
        num_arrays=2,
    ),
    side=st.sampled_from(["left", "right"]),
    num_positional_args_method=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.ndarray.searchsorted"
    ),
)
def test_numpy_instance_searchsorted(
    dtype_x_v,
    side,
    as_variable,
    num_positional_args_method,
    native_array,
):
    input_dtype, xs = dtype_x_v

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
            "v": xs[1],
            "side": side,
            "sorter": np.argsort(xs[0]),
        },
        frontend="numpy",
        class_name="ndarray",
        method_name="searchsorted",
    )


@handle_cmd_line_args
@given(
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
    num_positional_args_method=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.ndarray.squeeze"
    ),
)
def test_numpy_instance_squeeze(
    dtype_x_axis,
    as_variable,
    num_positional_args_method,
    native_array,
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
        },
        frontend="numpy",
        class_name="ndarray",
        method_name="squeeze",
    )


@handle_cmd_line_args
@given(
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
        frontend="numpy",
        class_name="ndarray",
        method_name="__add__",
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"), num_arrays=2
    ),
    num_positional_args_method=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.ndarray.__sub__"
    ),
)
def test_numpy_instance_sub__(
    dtype_and_x,
    as_variable,
    num_positional_args_method,
    native_array,
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
        frontend="numpy",
        class_name="ndarray",
        method_name="__sub__",
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
    num_positional_args_method=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.ndarray.__mul__"
    ),
)
def test_numpy_instance_mul__(
    dtype_and_x,
    as_variable,
    num_positional_args_method,
    native_array,
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
        frontend="numpy",
        class_name="ndarray",
        method_name="__mul__",
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("bool"),
        num_arrays=2,
    ),
    num_positional_args_method=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.ndarray.__and__"
    ),
)
def test_numpy_instance_and__(
    dtype_and_x,
    as_variable,
    num_positional_args_method,
    native_array,
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
        frontend="numpy",
        class_name="ndarray",
        method_name="__and__",
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("bool"),
        num_arrays=2,
    ),
    num_positional_args_method=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.ndarray.__or__"
    ),
)
def test_numpy_instance_or__(
    dtype_and_x,
    as_variable,
    num_positional_args_method,
    native_array,
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
        frontend="numpy",
        class_name="ndarray",
        method_name="__or__",
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("bool"),
        num_arrays=2,
    ),
    num_positional_args_method=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.ndarray.__xor__"
    ),
)
def test_numpy_instance_xor__(
    dtype_and_x,
    as_variable,
    num_positional_args_method,
    native_array,
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
        frontend="numpy",
        class_name="ndarray",
        method_name="__xor__",
    )


@handle_cmd_line_args
@given(
    x=_get_first_matrix_and_dtype(),
    y=_get_second_matrix_and_dtype(),
    num_positional_args_method=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.ndarray.__matmul__"
    ),
)
def test_numpy_instance_matmul__(
    x,
    y,
    as_variable,
    num_positional_args_method,
    native_array,
):
    dtype1, x1 = x
    dtype2, x2 = y
    input_dtype = dtype1 + dtype2

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
            "data": x1,
        },
        all_as_kwargs_np_method={
            "value": x2,
        },
        frontend="numpy",
        class_name="ndarray",
        method_name="__matmul__",
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
    ),
    num_positional_args_method=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.ndarray.__copy__"
    ),
)
def test_numpy_instance_copy__(
    dtype_and_x,
    as_variable,
    num_positional_args_method,
    native_array,
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
        all_as_kwargs_np_method={},
        frontend="numpy",
        class_name="ndarray",
        method_name="__copy__",
    )
