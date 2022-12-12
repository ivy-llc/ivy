# local
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# logical_and
@handle_frontend_test(
    fn_tree="numpy.logical_and",
    dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(
        arr_func=[
            lambda: helpers.dtype_and_values(
                available_dtypes=helpers.get_dtypes("bool"),
                num_arrays=2,
            )
        ],
        get_dtypes_kind="bool",
    ),
    where=np_frontend_helpers.where(),
)
def test_numpy_logical_and(
    dtypes_values_casting,
    where,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtypes, x, casting, dtype = dtypes_values_casting
    where, as_variable, native_array = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=dtypes,
        as_variable=as_variable,
        native_array=native_array,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=x[0],
        x2=x[1],
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
    dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(
        arr_func=[
            lambda: helpers.dtype_and_values(
                available_dtypes=helpers.get_dtypes("bool"),
                num_arrays=2,
            )
        ],
        get_dtypes_kind="bool",
    ),
    where=np_frontend_helpers.where(),
)
def test_numpy_logical_or(
    dtypes_values_casting,
    where,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtypes, x, casting, dtype = dtypes_values_casting
    where, as_variable, native_array = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=dtypes,
        as_variable=as_variable,
        native_array=native_array,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=x[0],
        x2=x[1],
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
    dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(
        arr_func=[
            lambda: helpers.dtype_and_values(
                available_dtypes=helpers.get_dtypes("bool"),
                num_arrays=2,
            )
        ],
        get_dtypes_kind="bool",
    ),
    where=np_frontend_helpers.where(),
)
def test_numpy_logical_not(
    dtypes_values_casting,
    where,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtypes, x, casting, dtype = dtypes_values_casting
    where, as_variable, native_array = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=dtypes,
        as_variable=as_variable,
        native_array=native_array,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=dtypes,
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
    dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(
        arr_func=[
            lambda: helpers.dtype_and_values(
                available_dtypes=helpers.get_dtypes("bool"),
                num_arrays=2,
            )
        ],
        get_dtypes_kind="bool",
    ),
    where=np_frontend_helpers.where(),
)
def test_numpy_logical_xor(
    dtypes_values_casting,
    where,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtypes, x, casting, dtype = dtypes_values_casting
    where, as_variable, native_array = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=dtypes,
        as_variable=as_variable,
        native_array=native_array,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        x1=x[0],
        x2=x[1],
        out=None,
        where=where,
        casting=casting,
        order="K",
        dtype=dtype,
        subok=True,
    )
