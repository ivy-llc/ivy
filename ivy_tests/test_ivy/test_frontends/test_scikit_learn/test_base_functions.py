
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import assert_all_close
from ivy_tests.test_ivy.helpers import handle_frontend_test


@handle_frontend_test(
    fn_tree="scikit-learn.is_classifier",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def scikit_learn_is_classifier(
        *,
        dtype_and_x,
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
        x=x[0],
    )
