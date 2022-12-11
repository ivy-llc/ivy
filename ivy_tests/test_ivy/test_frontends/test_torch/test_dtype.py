# global
from hypothesis import settings

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers.testing_helpers import handle_frontend_test


# can_cast
@handle_frontend_test(
    fn_tree="torch.can_cast",
    from_=helpers.get_dtypes("valid", full=False),
    to=helpers.get_dtypes("valid", full=False),
)
# there are 100 combinations of dtypes, so run 200 examples to make sure all are tested
@settings(max_examples=200)
def test_torch_can_cast(
    *,
    from_,
    to,
    as_variable,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    helpers.test_frontend_function(
        input_dtypes=[],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=2,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        from_=ivy.Dtype(from_[0]),
        to=ivy.Dtype(to[0]),
    )
