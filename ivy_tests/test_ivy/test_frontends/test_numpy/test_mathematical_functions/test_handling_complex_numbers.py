# global
from hypothesis import strategies as st

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
    fn_tree = "numpy.conj",
    dtype_and_x = helpers.dtype_and_values(
        available_dtypes = helpers.get_dtypes("complex")
    ),
    test_with_out = st.just(False),
)
def test_numpy_conj(
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    out,
    where,
    casting,
    order,
    dtype,
    subok,
    signature,
    extobj,
):
    input_dtypes, x = dtype_and_x
    kwargs = {
        "out": out,
        "where": where,
        "casting": casting,
        "order": order,
        "dtype": dtype,
        "subok": subok,
        "signature": signature,
        "extobj": extobj,
    }
    helpers.test_frontend_function(
        input_dtypes = input_dtypes,
        frontend = frontend,
        test_flags = test_flags,
        fn_tree = fn_tree,
        on_device = on_device,
        val = x[0],
        **kwargs
    )
