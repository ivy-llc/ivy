from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


@handle_frontend_test(
    fn_tree="jax.numpy.array",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
    dtype=helpers.get_dtypes("numeric", full=False, none=True),
    copy=st.booleans(),
    ndmin=helpers.ints(min_value=0, max_value=10),
)
def test_jax_numpy_array(
    *,
    dtype_and_x,
    dtype,
    copy,
    ndmin,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        object=x[0],
        dtype=dtype[0],
        copy=copy,
        order="K",
        ndmin=ndmin,
    )
