# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# sum
@handle_frontend_test(
    fn_tree="numpy.sum",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float")
    ),
    dtype=helpers.get_dtypes("float", full=False, none=True),
    keep_dims=st.booleans(),
    initial=st.one_of(st.floats(), st.none()),
    where=np_frontend_helpers.where(),
)
def test_numpy_sum(
    dtype_x_axis,
    dtype,
    keep_dims,
    where,
    initial,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x, axis = dtype_x_axis
    if initial is not None:
        (
            where,
            as_variable,
            test_flags.native_arrays,
        ) = np_frontend_helpers.handle_where_and_array_bools(
            where=where,
            input_dtype=input_dtype,
            as_variable=test_flags.as_variable,
            native_array=test_flags.native_arrays,
        )
    else:
        where = None
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        axis=axis,
        dtype=dtype[0],
        keepdims=keep_dims,
        initial=initial,
        where=where,
    )


# prod
@handle_frontend_test(
    fn_tree="numpy.prod",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float")
    ),
    dtype=helpers.get_dtypes("float", full=False, none=True),
    keep_dims=st.booleans(),
    initial=st.one_of(st.floats(), st.none()),
    where=np_frontend_helpers.where(),
)
def test_numpy_prod(
    dtype_x_axis,
    dtype,
    keep_dims,
    initial,
    where,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x, axis = dtype_x_axis
    if initial is not None:
        (
            where,
            as_variable,
            test_flags.native_arrays,
        ) = np_frontend_helpers.handle_where_and_array_bools(
            where=where,
            input_dtype=input_dtype,
            as_variable=test_flags.as_variable,
            native_array=test_flags.native_arrays,
        )
    else:
        where = None
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        axis=axis,
        dtype=dtype[0],
        keepdims=keep_dims,
        initial=initial,
        where=where,
    )


# cumsum
@handle_frontend_test(
    fn_tree="numpy.cumsum",
    dtype_and_x=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
    ),
    dtype=helpers.get_dtypes("float", full=False, none=True),
)
def test_numpy_cumsum(
    dtype_and_x,
    dtype,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x, axis = dtype_and_x
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        axis=axis,
        dtype=dtype[0],
    )


# cumprod
@handle_frontend_test(
    fn_tree="numpy.cumprod",
    dtype_and_x=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
    ),
    dtype=helpers.get_dtypes("float", full=False, none=True),
)
def test_numpy_cumprod(
    dtype_and_x,
    dtype,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x, axis = dtype_and_x
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        axis=axis,
        dtype=dtype[0],
    )


# nancumprod
@handle_frontend_test(
    fn_tree="numpy.nancumprod",
    dtype_and_x=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
    ),
    dtype=helpers.get_dtypes("float", full=False, none=True),
)
def test_numpy_nancumprod(
    dtype_and_x,
    dtype,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x, axis = dtype_and_x
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        axis=axis,
        dtype=dtype[0],
    )


# nancumsum
@handle_frontend_test(
    fn_tree="numpy.nancumsum",
    dtype_and_x=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
    ),
    dtype=helpers.get_dtypes("float", full=False, none=True),
)
def test_numpy_nancumsum(
    dtype_and_x,
    dtype,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x, axis = dtype_and_x
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        axis=axis,
        dtype=dtype[0],
    )


# nanprod
@handle_frontend_test(
    fn_tree="numpy.nanprod",
    dtype_and_x=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
        large_abs_safety_factor=2,
        safety_factor_scale="log",
    ),
    dtype=helpers.get_dtypes("float", full=False, none=True),
    keepdims=st.booleans(),
    where=np_frontend_helpers.where(),
)
def test_numpy_nanprod(
    dtype_and_x,
    dtype,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    where,
    keepdims,
):
    input_dtype, x, axis = dtype_and_x
    where, as_variable, native_array = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtype,
        as_variable=test_flags.as_variable,
        native_array=test_flags.native_arrays,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis=axis,
        dtype=dtype[0],
        where=where,
        keepdims=keepdims,
    )


# nansum
@handle_frontend_test(
    fn_tree="numpy.nansum",
    dtype_and_x=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
        large_abs_safety_factor=2,
        safety_factor_scale="log",
    ),
    dtype=helpers.get_dtypes("float", full=False, none=True),
    keepdims=st.booleans(),
    where=np_frontend_helpers.where(),
)
def test_numpy_nansum(
    dtype_and_x,
    dtype,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    where,
    keepdims,
):
    input_dtype, x, axis = dtype_and_x
    where, as_variable, native_array = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtype,
        as_variable=test_flags.as_variable,
        native_array=test_flags.native_arrays,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis=axis,
        dtype=dtype[0],
        where=where,
        keepdims=keepdims,
    )
