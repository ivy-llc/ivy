# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args
import ivy


# mean
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.statistical_dtype_values(function="mean"),
    dtype=helpers.get_dtypes("float", full=False, none=True),
    where=np_frontend_helpers.where(),
    as_variable=helpers.array_bools(num_arrays=1),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.mean"
    ),
    native_array=helpers.array_bools(num_arrays=1),
    keep_dims=st.booleans(),
)
def test_numpy_mean(
    dtype_and_x,
    dtype,
    where,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
    keep_dims,
):
    input_dtype, x, axis = dtype_and_x
    x_array = ivy.array(x)

    if len(x_array.shape) == 2:
        where = ivy.ones((x_array.shape[0], 1), dtype=ivy.bool)
    elif len(x_array.shape) == 1:
        where = True

    if isinstance(axis, tuple):
        axis = axis[0]

    input_dtype = [input_dtype]
    where = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtype,
        as_variable=as_variable,
        native_array=native_array,
    )

    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="mean",
        x=np.asarray(x, dtype=input_dtype[0]),
        axis=axis,
        dtype=dtype,
        out=None,
        keepdims=keep_dims,
        where=where,
        test_values=False,
    )
