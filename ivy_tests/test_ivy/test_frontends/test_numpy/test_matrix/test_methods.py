# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@st.composite
def _array_with_dtype_axis_keepdims_and_where(draw):
    dtypes = draw(helpers.array_dtypes(num_arrays=1))
    shape = draw(
        helpers.get_shape(
            min_num_dims=1,
            max_num_dims=5,
        )
    )
    axis = draw(helpers.ints(min_value=-1, max_value=len(shape) - 1))
    if axis == -1:
        axis = None
    x = draw(
        helpers.array_values(
            shape=shape,
            dtype=dtypes[0],
        )
    )
    where_shape_length = draw(helpers.ints(min_value=0, max_value=len(shape)))
    if where_shape_length != 0:
        where_nb_dims_to_change = draw(
            helpers.ints(
                min_value=0,
                max_value=where_shape_length - 1,
            )
        )
        where_dims_to_change = [
            draw(
                helpers.ints(
                    min_value=0,
                    max_value=where_shape_length - 1,
                )
            )
            for i in range(where_nb_dims_to_change)
        ]
        where_dims_list = [1] * where_shape_length
        for dim in where_dims_to_change:
            where_dims_list[dim] = shape[::-1][dim]
        where_dims_list = where_dims_list[::-1]
        where = draw(
            helpers.array_values(
                shape=where_dims_list,
                dtype="bool",
            )
        )
    else:
        where = draw(st.booleans())
    keepdims = draw(st.booleans())
    return x, dtypes[0], axis, keepdims, where


# argmax
@handle_cmd_line_args
@given(
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        min_dim_size=1,
        valid_axis=True,
        force_int_axis=True,
        allow_neg_axes=False,
    ),
    as_variable=helpers.array_bools(num_arrays=1),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.argmax"
    ),
    native_array=helpers.array_bools(num_arrays=1),
    keep_dims=st.booleans(),
)
def test_numpy_argmax(
    dtype_x_axis,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
    keep_dims,
):
    input_dtype, x, axis = dtype_x_axis
    if isinstance(axis, tuple):
        axis = axis[0]
    input_dtype = [input_dtype]
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="argmax",
        a=np.asarray(x, dtype=input_dtype[0]),
        axis=axis,
        out=None,
        keepdims=keep_dims,
        test_values=False,
    )


# any
@given(
    x_dtype_axis_keepdims_where=_array_with_dtype_axis_keepdims_and_where(),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.any"
    ),
    native_array=st.booleans(),
)
def test_numpy_any(
    x_dtype_axis_keepdims_where,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    x, input_dtype, axis, keepdims, where = x_dtype_axis_keepdims_where
    x = np.asarray(x, dtype=input_dtype)
    num_positional_args = num_positional_args
    input_dtypes = [input_dtype]
    as_variable = [as_variable]
    native_array = [native_array]
    if keepdims:
        out = (
            ivy.zeros(
                [
                    dim if i != axis and axis is not None else 1
                    for i, dim in enumerate(x.shape)
                ],
                dtype=bool,
            )
            if with_out
            else None
        )
    else:
        out = (
            ivy.zeros(
                [
                    dim
                    for i, dim in enumerate(x.shape)
                    if i != axis and axis is not None
                ],
                dtype=bool,
            )
            if with_out
            else None
        )
    where = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        as_variable=as_variable,
        native_array=native_array,
    )

    if fw == "torch":
        keepdims = True
    ret, ret_gt = helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="any",
        x=x,
        axis=axis,
        out=out,
        keepdims=keepdims,
        where=where,
        test_values=False,
    )

    if isinstance(ret_gt, tuple):
        ret_gt = ret_gt[0]
    if len(ret.shape) == 0:
        assert ret and ret_gt or not ret and not ret_gt
    else:
        assert ret.shape == ret_gt.shape and np.all(ret == ret_gt)
