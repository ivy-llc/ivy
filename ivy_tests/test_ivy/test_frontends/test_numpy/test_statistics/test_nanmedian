# global
import ivy

# local
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers


def test_numpy_nanmedian(
    dtype_and_a,
    dtype,
    where,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    keep_dims,
):
    input_dtype, a, axis = dtype_and_a
    a_array = ivy.array(a[0])

    if len(a_array.shape) == 2:
        where = ivy.ones((a_array.shape[0], 1), dtype=ivy.bool)
    elif len(a_array.shape) == 1:
        where = True
    if isinstance(axis, tuple):
        axis = axis[0]
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
        frontend="numpy",
        fn_tree="nanmedian",
        a=a[0],
        axis=axis,
        dtype=dtype,
        out=None,
        keepdims=keep_dims,
        where=where,
        test_values=False,
    )
