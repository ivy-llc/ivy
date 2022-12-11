# global

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


@handle_frontend_test(
    fn_tree="numpy.isfinite",
    dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(
        arr_func=[
            lambda: helpers.dtype_and_values(
                available_dtypes=helpers.get_dtypes("float"),
            )
        ],
        special=True,
    ),
    where=np_frontend_helpers.where(),
)
def test_numpy_isfinite(
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


@handle_frontend_test(
    fn_tree="numpy.isinf",
    dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(
        arr_func=[
            lambda: helpers.dtype_and_values(
                available_dtypes=helpers.get_dtypes("float"),
            )
        ],
        special=True,
    ),
    where=np_frontend_helpers.where(),
)
def test_numpy_isinf(
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


@handle_frontend_test(
    fn_tree="numpy.isnan",
    dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(
        arr_func=[
            lambda: helpers.dtype_and_values(
                available_dtypes=helpers.get_dtypes("float"),
            )
        ],
        special=True,
    ),
    where=np_frontend_helpers.where(),
)
def test_numpy_isnan(
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
