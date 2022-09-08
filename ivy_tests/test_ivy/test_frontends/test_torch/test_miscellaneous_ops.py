# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
import ivy.functional.backends.torch as ivy_torch
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# flip
@handle_cmd_line_args
@given(
    dtype_and_values=helpers.dtype_and_values(
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(
                set(ivy_torch.valid_float_dtypes)
            ),
        ),
    ),
    axis=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
        force_tuple=True,
    ),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.flip"
    ),
    native_array=st.booleans(),
)
def test_torch_flip(
    dtype_and_values,
    axis,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, value = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="flip",
        input=np.asarray(value, dtype=input_dtype),
        dims=axis,
    )


# roll
@given(
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(
                set(ivy_torch.valid_float_dtypes)
            )
        ),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ),
    shift=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ),
    axis=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.roll"
    ),
    native_array=st.booleans(),
)
def test_torch_roll(
    dtype_and_values,
    shift,
    axis,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, value = dtype_and_values
    if isinstance(shift, int) and isinstance(axis, tuple):
        axis = axis[0]
    if isinstance(shift, tuple) and isinstance(axis, tuple):
        if len(shift) != len(axis):
            mn = min(len(shift), len(axis))
            shift = shift[:mn]
            axis = axis[:mn]
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="roll",
        input=np.asarray(value, dtype=input_dtype),
        shifts=shift,
        dims=axis,
    )


# fliplr
@given(
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(
                set(ivy_torch.valid_float_dtypes)
            ),
        ),
        shape=helpers.get_shape(min_num_dims=2),
    ),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.fliplr"
    ),
    native_array=st.booleans(),
)
def test_torch_fliplr(
    dtype_and_values,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, value = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="fliplr",
        input=np.asarray(value, dtype=input_dtype),
    )


# cumsum
@handle_cmd_line_args
@given(
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(
                set(ivy_torch.valid_float_dtypes)
            ),
        ),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ),
    axis=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape"),
    ).filter(lambda axis: isinstance(axis, int)),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.cumsum"
    ),
    native_array=st.booleans(),
)
def test_torch_cumsum(
    dtype_and_values,
    axis,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, value = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=True,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="cumsum",
        input=np.asarray(value, dtype=input_dtype),
        dim=axis,
        dtype=input_dtype,
        out=None,
    )


# TODO: Widen the bounds on the row, col, and offset data
#  once ivy.asarray has been optimized.
@handle_cmd_line_args
@given(
    dtype_and_size=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_int_dtypes).intersection(set(ivy_torch.valid_int_dtypes)),
        ),
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
        max_dim_size=2,
        min_value=0,
    ),
    dtype_and_offset=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_int_dtypes).intersection(set(ivy_torch.valid_int_dtypes)),
        ),
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=1,
        max_dim_size=1,
        num_arrays=1,
    ),
    dtype_result=helpers.get_dtypes(
        kind="integer",
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.tril_indices"
    ),
)
def test_torch_tril_indices(
    dtype_and_size,
    dtype_and_offset,
    dtype_result,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    dtype_row_col, size = dtype_and_size
    row, col = size

    dtype_offset, offset = dtype_and_offset
    offset = offset[
        0
    ]  # The reason for offset being a 2-list here completely evades me.

    helpers.test_frontend_function(
        input_dtypes=dtype_row_col + dtype_offset,
        with_out=False,
        num_positional_args=num_positional_args,
        as_variable_flags=as_variable,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="tril_indices",
        row=row,
        col=col,
        offset=offset,
        dtype=dtype_result,
    )
