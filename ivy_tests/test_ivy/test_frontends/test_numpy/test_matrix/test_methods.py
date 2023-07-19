# global
from hypothesis import assume, strategies as st
import numpy as np
import sys

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_method, handle_frontend_test
import ivy.functional.frontends.numpy as ivy_np


CLASS_TREE = "ivy.functional.frontends.numpy.matrix"


def _to_string_matrix(num_matrix):
    str_matrix = ""
    for i, row in enumerate(num_matrix):
        for j, elem in enumerate(row):
            str_matrix += str(elem)
            if j < num_matrix.shape[1] - 1:
                str_matrix += " "
            elif i < num_matrix.shape[0] - 1:
                str_matrix += "; "
    return str_matrix


def _get_x_matrix(x, to_str):
    if to_str:
        x = _to_string_matrix(x[0])
    else:
        x = x[0]
    return x


@st.composite
def _property_helper(draw):
    _, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"),
            min_value=-1e04,
            max_value=1e04,
            min_num_dims=2,
            max_num_dims=2,
        )
    )
    to_str = (st.booleans(),)
    x = _get_x_matrix(x, to_str)
    data = ivy_np.matrix(x)
    data_gt = np.matrix(x)
    return data, data_gt


@handle_frontend_test(
    fn_tree="numpy.add",  # dummy fn_tree
    matrices=_property_helper(),
)
def test_numpy_A(matrices):
    data, data_gt = matrices
    ret = np.ravel(data.A)
    ret_gt = np.ravel(data_gt.A)
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend="numpy",
    )


@handle_frontend_test(
    fn_tree="numpy.add",  # dummy fn_tree
    matrices=_property_helper(),
)
def test_numpy_A1(matrices):
    data, data_gt = matrices
    helpers.value_test(
        ret_np_flat=data.A1,
        ret_np_from_gt_flat=data_gt.A1,
        ground_truth_backend="numpy",
    )


@handle_frontend_test(
    fn_tree="numpy.add",  # dummy fn_tree
    matrices=_property_helper(),
)
def test_numpy_I(matrices):
    data, data_gt = matrices
    assume(
        np.linalg.cond(data.A.data) < 1 / sys.float_info.epsilon
        and data.shape[0] == data.shape[1]
    )
    ret = np.ravel(data.I)
    ret_gt = np.ravel(data_gt.I)
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend="numpy",
    )


@handle_frontend_test(
    fn_tree="numpy.add",  # dummy fn_tree
    matrices=_property_helper(),
)
def test_numpy_T(matrices):
    data, data_gt = matrices
    ret = np.ravel(data.T)
    ret_gt = np.ravel(data_gt.T)
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend="numpy",
    )


@handle_frontend_test(
    fn_tree="numpy.add",  # dummy fn_tree
    matrices=_property_helper(),
)
def test_numpy_data(matrices):
    data, data_gt = matrices
    # sanity test
    ivy.utils.assertions.check_equal(
        type(data.data), type(data_gt.data), as_array=False
    )


@handle_frontend_test(
    fn_tree="numpy.add",  # dummy fn_tree
    matrices=_property_helper(),
)
def test_numpy_dtype(matrices):
    data, data_gt = matrices
    ivy.utils.assertions.check_equal(
        str(data.dtype), str(data_gt.dtype), as_array=False
    )


@handle_frontend_test(
    fn_tree="numpy.add",  # dummy fn_tree
    matrices=_property_helper(),
)
def test_numpy_ndim(matrices):
    data, data_gt = matrices
    ivy.utils.assertions.check_equal(data.ndim, data_gt.ndim, as_array=False)


@handle_frontend_test(
    fn_tree="numpy.add",  # dummy fn_tree
    matrices=_property_helper(),
)
def test_numpy_shape(matrices):
    data, data_gt = matrices
    ivy.utils.assertions.check_equal(data.shape, data_gt.shape, as_array=False)


@handle_frontend_test(
    fn_tree="numpy.add",  # dummy fn_tree
    matrices=_property_helper(),
)
def test_numpy_size(matrices):
    data, data_gt = matrices
    ivy.utils.assertions.check_equal(data.size, data_gt.size, as_array=False)


# argmax
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.matrix",
    method_name="argmax",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=2,
        valid_axis=True,
        force_int_axis=True,
        allow_neg_axes=False,
    ),
    to_str=st.booleans(),
)
def test_numpy_matrix_argmax(
    dtype_x_axis,
    to_str,
    init_flags,
    method_flags,
    frontend_method_data,
    frontend,
    on_device,
):
    input_dtype, x, axis = dtype_x_axis
    x = _get_x_matrix(x, to_str)
    if isinstance(axis, tuple):
        axis = axis[0]
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={
            "data": x,
            "dtype": input_dtype[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "axis": axis,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        on_device=on_device,
    )


# any
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="numpy.matrix",
    method_name="any",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=2,
        valid_axis=True,
        force_int_axis=True,
        allow_neg_axes=False,
    ),
    to_str=st.booleans(),
)
def test_numpy_matrix_any(
    dtype_x_axis,
    to_str,
    init_flags,
    method_flags,
    frontend_method_data,
    frontend,
    on_device,
):
    input_dtype, x, axis = dtype_x_axis
    x = _get_x_matrix(x, to_str)
    if isinstance(axis, tuple):
        axis = axis[0]
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={
            "data": x,
            "dtype": input_dtype[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={
            "axis": axis,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        on_device=on_device,
    )
