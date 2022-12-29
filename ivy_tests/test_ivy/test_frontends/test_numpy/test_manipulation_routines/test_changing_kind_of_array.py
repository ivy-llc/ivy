# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from hypothesis import strategies as st


# asanyarray
@handle_frontend_test(
    fn_tree="numpy.asanyarray",
    dtype=helpers.get_dtypes("valid", full=False),
    order=st.sampled_from(["C", "F", "A", "K"]),
    like=helpers.get_dtypes("valid", full=False),
)
def test_numpy_asanyarray(
    a,
    dtype,
    order,
    *,
    like,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        a=a,
        dtype=dtype[0],
        order=order,
        like=like,
    )
