# global
from hypothesis import strategies as st, assume
import ivy

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# all
@handle_frontend_test(
    fn_tree="numpy.all",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        valid_axis=True,
        max_axes_size=1,
        force_int_axis=True,
    ),
    keepdims=st.booleans(),
    where=np_frontend_helpers.where(),
)
def test_numpy_all(
    *,
    dtype_x_axis,
    keepdims,
    where,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtypes, x, axis = dtype_x_axis
    axis = axis if axis is None or isinstance(axis, int) else axis[0]
    where, input_dtypes, test_flags = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis=axis,
        out=None,
        keepdims=keepdims,
        where=where,
    )


# any
@handle_frontend_test(
    fn_tree="numpy.any",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        valid_axis=True,
        max_axes_size=1,
        force_int_axis=True,
    ),
    keepdims=st.booleans(),
    where=np_frontend_helpers.where(),
)
def test_numpy_any(
    *,
    dtype_x_axis,
    keepdims,
    where,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtypes, x, axis = dtype_x_axis
    axis = axis if axis is None or isinstance(axis, int) else axis[0]
    where, input_dtypes, test_flags = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis=axis,
        out=None,
        keepdims=keepdims,
        where=where,
    )


@handle_frontend_test(
    fn_tree="numpy.isscalar",
    element=st.booleans() | st.floats() | st.integers() | st.complex_numbers(),
    test_with_out=st.just(False),
)
def test_numpy_isscalar(
    *,
    element,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    helpers.test_frontend_function(
        input_dtypes=ivy.all_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        element=element,
    )


@handle_frontend_test(
    fn_tree="numpy.isfortran",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"), min_num_dims=1
    ),
    test_with_out=st.just(False),
)
def test_numpy_isfortran(
    *,
    dtype_and_x,
    frontend,
    on_device,
    fn_tree,
    test_flags,
    backend_fw,
):
    if ivy.current_backend() != "numpy":
        return
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
    )


@handle_frontend_test(
    fn_tree="numpy.isreal",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_complex")
    ),
)
def test_numpy_isreal(
    *,
    dtype_and_x,
    frontend,
    on_device,
    fn_tree,
    test_flags,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


@handle_frontend_test(
    fn_tree="numpy.isrealobj",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("real_and_complex"), min_num_dims=1
    ),
    test_with_out=st.just(False),
)
def test_numpy_isrealobj(
    *,
    dtype_and_x,
    frontend,
    on_device,
    fn_tree,
    test_flags,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


@handle_frontend_test(
    fn_tree="numpy.iscomplexobj",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("real_and_complex"),
    ),
    test_with_out=st.just(False),
)
def test_numpy_iscomplexobj(
    *,
    dtype_and_x,
    frontend,
    on_device,
    fn_tree,
    test_flags,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    if ivy.current_backend_str() == "paddle":
        # mostly paddle doesn't support unsigned int
        assume(input_dtype[0] not in ["int8", "uint8", "int16"])
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


@handle_frontend_test(
    fn_tree="numpy.iscomplex",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("real_and_complex"), min_num_dims=1
    ),
    test_with_out=st.just(False),
)
def test_numpy_iscomplex(
    *,
    dtype_and_x,
    frontend,
    on_device,
    fn_tree,
    test_flags,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )
