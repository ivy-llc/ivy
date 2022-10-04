# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args, num_positional_args
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers


# argmax
@handle_cmd_line_args
given(
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
            "data":x[0],
        },
        input_dtypes_method=[input_dtype[1]],
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={
            "value":x[1],
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
            shape=helpers.get_shape(
                allow_none=False,
                min_num_dims=1,
                max_num_dims=5,
                min_dim_size=1,
                max_dim_size=10,
            )
        )
    )
    shape = draw(helpers.reshape_shapes(shape=np.array(x).shape))
    return dtypes, x, shape


@handle_cmd_line_args
@given(
    dtypes_x_shape=dtypes_x_reshape(),
    copy=st.booleans(),
    with_out=st.booleans(),
    as_variable=helpers.array_bools(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.ndarray.reshape"
    ),
    native_array=helpers.array_bools(),
)
def test_numpy_instance_reshape(
    dtypes_x_shape,
    copy,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    dtypes, x, shape = dtypes_x_shape
    helpers.test_frontend_array_instance_method(
        input_dtypes=dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        frontend_class=ndarray,
        fn_tree="ndarray.reshape",
        self=x,
        newshape=shape,
        copy=copy,
    )


# add
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_float_dtypes, num_arrays=2
    ),
    dtype=st.sampled_from(ivy_np.valid_float_dtypes + (None,)),
    where=np_frontend_helpers.where(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.ndarray.add"
    ),
)
def test_numpy_instance_add(
    dtype_and_x,
    dtype,
    where,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x
    where = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtype,
        as_variable=as_variable,
        native_array=native_array,
    )
    np_frontend_helpers.test_frontend_array_instance_method(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        frontend_class=ndarray,
        fn_tree="ndarray.add",
        self=np.asarray(x[0], dtype=input_dtype[0]),
        other=np.asarray(x[1], dtype=input_dtype[1]),
        out=None,
        where=where,
        casting="same_kind",
        order="k",
        dtype=dtype,
        subok=True,
        test_values=False,
    )
