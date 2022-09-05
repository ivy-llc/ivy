# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args
import ivy.functional.backends.numpy as ivy_np
import ivy.functional.backends.torch as ivy_torch


@handle_cmd_line_args
@given(
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=tuple(
            set(ivy_np.valid_numeric_dtypes).intersection(
                set(ivy_torch.valid_numeric_dtypes)
            ),
        ),
        min_axis=-1,
        max_axis=0,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.argmax"
    ),
    keepdims=st.booleans(),
)
def test_torch_argmax(
    dtype_input_axis,
    as_variable,
    num_positional_args,
    native_array,
    keepdims,
    fw,
):
    input_dtype, x, axis = dtype_input_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="argmax",
        input=np.asarray(x, dtype=input_dtype),
        dim=axis,
        keepdim=keepdims,
    )


@handle_cmd_line_args
@given(
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=tuple(
            set(ivy_np.valid_numeric_dtypes).intersection(
                set(ivy_torch.valid_numeric_dtypes)
            ),
        ),
        min_axis=-1,
        max_axis=0,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.argmin"
    ),
    keepdims=st.booleans(),
)
def test_torch_argmin(
    dtype_input_axis,
    as_variable,
    num_positional_args,
    native_array,
    keepdims,
    fw,
):
    input_dtype, x, axis = dtype_input_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="argmin",
        input=np.asarray(x, dtype=input_dtype),
        dim=axis,
        keepdim=keepdims,
    )


@handle_cmd_line_args
@given(
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=tuple(
            set(ivy_np.valid_numeric_dtypes).intersection(
                set(ivy_torch.valid_numeric_dtypes)
            ),
        ),
        min_axis=-1,
        max_axis=0,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.amax"
    ),
    keepdims=st.booleans(),
    with_out=st.booleans(),
)
def test_torch_amax(
    dtype_input_axis,
    as_variable,
    num_positional_args,
    native_array,
    keepdims,
    with_out,
    fw,
):
    input_dtype, x, axis = dtype_input_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="amax",
        input=np.asarray(x, dtype=input_dtype),
        dim=axis,
        keepdim=keepdims,
        out=None,
    )


@handle_cmd_line_args
@given(
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=tuple(
            set(ivy_np.valid_numeric_dtypes).intersection(
                set(ivy_torch.valid_numeric_dtypes)
            ),
        ),
        min_axis=-1,
        max_axis=0,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.amin"
    ),
    keepdims=st.booleans(),
    with_out=st.booleans(),
)
def test_torch_amin(
    dtype_input_axis,
    as_variable,
    num_positional_args,
    native_array,
    keepdims,
    with_out,
    fw,
):
    input_dtype, x, axis = dtype_input_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="amin",
        input=np.asarray(x, dtype=input_dtype),
        dim=axis,
        keepdim=keepdims,
        out=None,
    )


@handle_cmd_line_args
@given(
    dtypes_arrays=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_numeric_dtypes).intersection(
                set(ivy_torch.valid_numeric_dtypes)
            ),
        ),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
        num_arrays=3,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.aminmax"
    ),
    keepdims=st.booleans(),
    with_out=st.booleans(),
    axis=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
        min_size=1,
        max_size=1,
        force_int=True,
    ),
)
def test_torch_aminmax(
    dtypes_arrays,
    as_variable,
    num_positional_args,
    native_array,
    keepdims,
    with_out,
    axis,
    fw,
):
    input_dtype, x = dtypes_arrays
    out = None
    if with_out:
        if fw == "numpy" and len(x[0]) == 1:
            out = tuple(
                [
                    np.asarray(x[1][0], dtype=input_dtype[0]),
                    np.asarray(x[2][0], dtype=input_dtype[0]),
                ]
            )
        else:
            out = tuple(
                [
                    np.asarray(x[1], dtype=input_dtype[0]),
                    np.asarray(x[2], dtype=input_dtype[0]),
                ]
            )

    input_dtypes = input_dtype[0] if not with_out else input_dtype
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="aminmax",
        input=np.asarray(x[0], dtype=input_dtype[0]),
        dim=axis,
        keepdim=keepdims,
        out=out,
    )


@handle_cmd_line_args
@given(
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=tuple(
            set(ivy_np.valid_numeric_dtypes).intersection(
                set(ivy_torch.valid_numeric_dtypes)
            ),
        ),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        allow_inf=False,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.all"
    ),
    keepdims=st.booleans(),
    with_out=st.booleans(),
)
def test_torch_all(
    dtype_input_axis,
    as_variable,
    num_positional_args,
    native_array,
    keepdims,
    with_out,
    fw,
):
    input_dtype, x, axis = dtype_input_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="all",
        input=np.asarray(x, dtype=input_dtype),
        dim=axis,
        keepdim=keepdims,
        out=None,
    )


@handle_cmd_line_args
@given(
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=tuple(
            set(ivy_np.valid_numeric_dtypes).intersection(
                set(ivy_torch.valid_numeric_dtypes)
            ),
        ),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        allow_inf=False,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.any"
    ),
    keepdims=st.booleans(),
    with_out=st.booleans(),
)
def test_torch_any(
    dtype_input_axis,
    as_variable,
    num_positional_args,
    native_array,
    keepdims,
    with_out,
    fw,
):
    input_dtype, x, axis = dtype_input_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="any",
        input=np.asarray(x, dtype=input_dtype),
        dim=axis,
        keepdim=keepdims,
        out=None,
    )


@handle_cmd_line_args
@given(
    dtypes_arrays=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_numeric_dtypes).intersection(
                set(ivy_torch.valid_numeric_dtypes)
            ),
        ),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
        num_arrays=3,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.max"
    ),
    keepdims=st.booleans(),
    with_out=st.booleans(),
    axis=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
        min_size=1,
        max_size=1,
        force_int=True,
    ),
)
def test_torch_max(
    dtypes_arrays,
    as_variable,
    num_positional_args,
    native_array,
    keepdims,
    with_out,
    axis,
    fw,
):
    input_dtype, x = dtypes_arrays
    out = None
    if with_out:
        if fw == "numpy" and len(x[0]) == 1:
            out = tuple(
                [
                    np.asarray(x[1][0], dtype=input_dtype[0]),
                    np.asarray(x[2][0], dtype=input_dtype[0]),
                ]
            )
        else:
            out = tuple(
                [
                    np.asarray(x[1], dtype=input_dtype[0]),
                    np.asarray(x[2], dtype=input_dtype[0]),
                ]
            )

    input_dtypes = input_dtype[0] if not with_out else input_dtype
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="max",
        input=np.asarray(x[0], dtype=input_dtype[0]),
        dim=axis,
        keepdim=keepdims,
        out=out,
    )


@handle_cmd_line_args
@given(
    dtypes_arrays=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_numeric_dtypes).intersection(
                set(ivy_torch.valid_numeric_dtypes)
            ),
        ),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
        num_arrays=3,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.min"
    ),
    keepdims=st.booleans(),
    with_out=st.booleans(),
    axis=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
        min_size=1,
        max_size=1,
        force_int=True,
    ),
)
def test_torch_min(
    dtypes_arrays,
    as_variable,
    num_positional_args,
    native_array,
    keepdims,
    with_out,
    axis,
    fw,
):
    input_dtype, x = dtypes_arrays
    out = None
    if with_out:
        if fw == "numpy" and len(x[0]) == 1:
            out = tuple(
                [
                    np.asarray(x[1][0], dtype=input_dtype[0]),
                    np.asarray(x[2][0], dtype=input_dtype[0]),
                ]
            )
        else:
            out = tuple(
                [
                    np.asarray(x[1], dtype=input_dtype[0]),
                    np.asarray(x[2], dtype=input_dtype[0]),
                ]
            )

    input_dtypes = input_dtype[0] if not with_out else input_dtype
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="min",
        input=np.asarray(x[0], dtype=input_dtype[0]),
        dim=axis,
        keepdim=keepdims,
        out=out,
    )
