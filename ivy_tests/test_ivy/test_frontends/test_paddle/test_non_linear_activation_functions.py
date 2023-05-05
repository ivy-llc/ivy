# global
import ivy
from hypothesis import assume, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# tanh
@handle_frontend_test(
    fn_tree="torch.nn.functional.tanh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_tanh(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        atol=1e-2,
        input=x[0],
    )