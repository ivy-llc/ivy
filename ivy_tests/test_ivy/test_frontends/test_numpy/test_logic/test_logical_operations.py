# local
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# logical_and
@handle_frontend_test(
    fn_tree="numpy.logical_and",
    dtype_values_and_casting=np_frontend_helpers.get_dtype_and_values_and_casting(
        available_dtypes=("bool",),
        num_arrays=2,
        get_dtypes_kind="bool",
        get_dtypes_none=False,
    ),
    where=np_frontend_helpers.where(),
)
def test_numpy_logical_and(
    *,
    dtype_values_and_casting,
    where,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtype, input_dtype, xs, casting = dtype_values_and_casting
    where, as_variable, native_array = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtype[0],
        as_variable=as_variable,
        native_array=native_array,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=xs[0],
        x2=xs[1],
        out=None,
        where=where,
        casting=casting,
        order="K",
        dtype=dtype,
        subok=True,
    )


# logical_or
@handle_frontend_test(
    fn_tree="numpy.logical_or",
    dtype_values_and_casting=np_frontend_helpers.get_dtype_and_values_and_casting(
        available_dtypes=("bool",),
        num_arrays=2,
        get_dtypes_kind="bool",
        get_dtypes_none=False,
    ),
    where=np_frontend_helpers.where(),
)
def test_numpy_logical_or(
    *,
    dtype_values_and_casting,
    where,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtype, input_dtype, xs, casting = dtype_values_and_casting
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
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=xs[0],
        x2=xs[1],
        out=None,
        where=where,
        casting=casting,
        order="K",
        dtype=dtype,
        subok=True,
    )


# logical_not
@handle_frontend_test(
    fn_tree="numpy.logical_not",
    dtype_values_and_casting=np_frontend_helpers.get_dtype_and_values_and_casting(
        available_dtypes=("bool",),
        num_arrays=2,
        get_dtypes_kind="bool",
        get_dtypes_none=False,
    ),
    where=np_frontend_helpers.where(),
)
def test_numpy_logical_not(
    *,
    dtype_values_and_casting,
    where,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtype, input_dtype, x, casting = dtype_values_and_casting
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
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        out=None,
        where=where,
        casting=casting,
        order="K",
        dtype=dtype,
        subok=True,
    )


# logical_xor
@handle_frontend_test(
    fn_tree="numpy.logical_xor",
    dtype_values_and_casting=np_frontend_helpers.get_dtype_and_values_and_casting(
        available_dtypes=("bool",),
        num_arrays=2,
        get_dtypes_kind="bool",
        get_dtypes_none=False,
    ),
    where=np_frontend_helpers.where(),
)
def test_numpy_logical_xor(
    *,
    dtype_values_and_casting,
    where,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtype, input_dtype, xs, casting = dtype_values_and_casting
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
        frontend=frontend,
        fn_tree=fn_tree,
        x1=xs[0],
        x2=xs[1],
        out=None,
        where=where,
        casting=casting,
        order="K",
        dtype=dtype,
        subok=True,
    )
