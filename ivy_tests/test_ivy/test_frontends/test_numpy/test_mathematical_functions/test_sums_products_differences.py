# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# prod
@handle_cmd_line_args
@given(
    dtype_and_x=np_frontend_helpers.dtype_x_bounded_axis(
        available_dtypes=ivy.current_backend().valid_dtypes
    ),
    dtype=st.sampled_from(ivy.current_backend().valid_dtypes + (None,)),
    where=np_frontend_helpers.where(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.prod"
    ),
    keepdims=st.booleans(),
    initial=st.one_of(st.booleans(), st.integers(), st.floats(), st.complex_numbers()),
)
def test_numpy_prod(
    dtype_and_x,
    dtype,
    where,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
    keepdims,
    initial,
):
    input_dtype, x, axis = dtype_and_x
    input_dtype = [input_dtype]
    where = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtype,
        as_variable=[as_variable],
        native_array=[native_array],
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="prod",
        x=np.asarray(x, dtype=input_dtype[0]),
        axis=axis,
        dtype=dtype,
        out=None,
        keepdims=keepdims,
        initial=initial,
        where=where,
        test_values=True,
    )


@handle_cmd_line_args
@given(
    dtype_and_x=np_frontend_helpers.dtype_x_bounded_axis(
        available_dtypes=helpers.get_dtypes("valid", full=True),
        min_num_dims=1,
    ),
    dtype=helpers.get_dtypes("numeric"),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.cumsum"
    ),
)
def test_numpy_cumsum(
    dtype_and_x,
    dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x, axis = dtype_and_x
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="cumsum",
        x=np.asarray(x, dtype=input_dtype),
        axis=axis,
        dtype=None,
        out=None,
    )


@handle_cmd_line_args
@given(
    dtype_and_x=np_frontend_helpers.dtype_x_bounded_axis(
        available_dtypes=helpers.get_dtypes("valid", full=True),
        min_num_dims=1,
    ),
    dtype=helpers.get_dtypes("numeric"),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.cumprod"
    ),
)
def test_numpy_cumprod(
    dtype_and_x,
    dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x, axis = dtype_and_x
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="cumprod",
        x=np.asarray(x, dtype=input_dtype),
        axis=axis,
        dtype=None,
        out=None,
    )
