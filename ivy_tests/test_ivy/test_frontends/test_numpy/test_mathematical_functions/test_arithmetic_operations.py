# global
from hypothesis import given

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# add
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"), num_arrays=2
    ),
    dtype=helpers.get_dtypes("numeric", full=False, none=True),
    where=np_frontend_helpers.where(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.add"
    ),
)
def test_numpy_add(
    dtype_and_x,
    dtype,
    where,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtypes, xs = dtype_and_x
    where, as_variable, native_array = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        as_variable=as_variable,
        native_array=native_array,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="add",
        x1=xs[0],
        x2=xs[1],
        out=None,
        where=where,
        dtype=dtype,
    )


# subtract
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"), num_arrays=2
    ),
    dtype=helpers.get_dtypes("numeric", full=False, none=True),
    where=np_frontend_helpers.where(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.subtract"
    ),
)
def test_numpy_subtract(
    dtype_and_x,
    dtype,
    where,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtypes, xs = dtype_and_x
    where, as_variable, native_array = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        as_variable=as_variable,
        native_array=native_array,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="subtract",
        x1=xs[0],
        x2=xs[1],
        out=None,
        where=where,
        dtype=dtype,
    )


# vdot
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"), num_arrays=2
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.vdot"
    ),
)
def test_numpy_vdot(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtypes, xs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="vdot",
        a=xs[0],
        b=xs[1],
        test_values=False,
    )


# divide
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"), num_arrays=2
    ),
    dtype=helpers.get_dtypes("numeric", full=False, none=True),
    where=np_frontend_helpers.where(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.divide"
    ),
)
def test_numpy_divide(
    dtype_and_x,
    dtype,
    where,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtypes, xs = dtype_and_x
    where, as_variable, native_array = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        as_variable=as_variable,
        native_array=native_array,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="divide",
        x1=xs[0],
        x2=xs[1],
        out=None,
        where=where,
        dtype=dtype,
    )


# multiply
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"), num_arrays=2
    ),
    dtype=helpers.get_dtypes("numeric", full=False, none=True),
    where=np_frontend_helpers.where(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.multiply"
    ),
)
def test_numpy_multiply(
    dtype_and_x,
    dtype,
    where,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtypes, xs = dtype_and_x
    where, as_variable, native_array = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        as_variable=as_variable,
        native_array=native_array,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="multiply",
        x1=xs[0],
        x2=xs[1],
        out=None,
        where=where,
        dtype=dtype,
        subok=True,
    )


# positive
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
    ),
    dtype=helpers.get_dtypes("float", full=False, none=True),
    where=np_frontend_helpers.where(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.positive"
    ),
)
def test_numpy_positive(
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
    where, as_variable, native_array = np_frontend_helpers.handle_where_and_array_bools(
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
        fn_tree="positive",
        x=x[0],
        out=None,
        where=where,
        dtype=dtype,
    )


# negative
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"), min_num_dims=1
    ),
    dtype=helpers.get_dtypes("numeric", full=False, none=True),
    where=np_frontend_helpers.where(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.negative"
    ),
)
def test_numpy_negative(
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
    where, as_variable, native_array = np_frontend_helpers.handle_where_and_array_bools(
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
        fn_tree="negative",
        x=x[0],
        out=None,
        casting="unsafe",
        where=where,
        dtype=dtype,
    )
