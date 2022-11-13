# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


@handle_frontend_test(
    fn_tree="tensorflow.nest.flatten",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    expand_composite=st.booleans(),
    use_array=st.booleans(),
)
def test_tensorflow_flatten(
    *,
    dtype_and_x,
    expand_composite,
    use_array,
    as_variable,
    native_array,
    frontend,
    fn_tree,
    on_device,
    num_positional_args,
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
        structure=x[0] if use_array else x[0].tolist(),
        expand_composites=expand_composite,
    )
