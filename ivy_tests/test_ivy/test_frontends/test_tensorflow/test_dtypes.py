# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# as_dtype
@handle_frontend_test(
    fn_tree="tensorflow.as_dtype",
    input_dtype=helpers.get_dtypes("valid", full=False),
    test_with_out=st.just(False),
)
def test_tensorflow_as_dtype(
    *,
    input_dtype,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        type_value=input_dtype[0],
    )


# cast
@handle_frontend_test(
    fn_tree="tensorflow.cast",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    dtype=helpers.get_dtypes("valid"),
    test_with_out=st.just(False),
)
def test_tensorflow_cast(
    *,
    dtype_and_x,
    dtype,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype + dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        dtype=dtype[0],
    )
