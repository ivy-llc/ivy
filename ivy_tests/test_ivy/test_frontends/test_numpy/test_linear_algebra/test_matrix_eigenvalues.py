# global
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_tests.test_ivy.test_functional.test_core.test_linalg import (
    _get_dtype_and_matrix,
)


# eigvalsh
@handle_frontend_test(
    fn_tree="numpy.linalg.eigvalsh",
    x=_get_dtype_and_matrix(symmetric=True)
)
def test_numpy_eigvalsh(
    x,
    as_variable,
    with_out,
    native_array,
    num_positional_args,
    frontend,
    fn_tree,
    on_device,
):
    input_dtypes, xs = x
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="numpy",
        fn_tree=fn_tree,
        on_device=on_device,
        x=xs
    )
