# global
from hypothesis import strategies as st
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# angle
@handle_frontend_test(
    fn_tree="numpy.angle",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("complex")
    ),
    deg=st.booleans(),
)
def test_numpy_angle(
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    deg,
):
    input_dtypes, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        z=x[0],
        deg=deg,
    )


# imag
@handle_frontend_test(
    fn_tree="numpy.imag",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
    test_with_out=st.just(False),
)
def test_numpy_imag(
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtypes, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        val=x[0],
    )


# real
@handle_frontend_test(
    fn_tree="numpy.real",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
    test_with_out=st.just(False),
)
def test_numpy_real(
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtypes, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        val=x[0],
    )


# conj
@handle_frontend_test(
    fn_tree="numpy.conj",
    dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(
        arr_func=[
            lambda: helpers.dtype_and_values(
                available_dtypes=helpers.get_dtypes("complex"),
            )
        ],
        get_dtypes_kind="complex",
    ),
    where=np_frontend_helpers.where(),
    number_positional_args = np_frontend_helpers.get_num_positional_args_ufunc(
        fn_name="conj"
    ),
)
def test_numpy_conj(
    dtypes_values_casting,
    where,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtypes, x, casting, dtype = dtypes_values_casting
    where, input_dtypes, test_flags = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x,
        out=None,
        where=where,
        casting=casting,
        order="K",
        dtype=dtype,
        subok=True,
    )
