# global
from hypothesis import settings, strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers.testing_helpers import handle_frontend_test
import ivy.functional.frontends.torch as torch_frontend


# can_cast
@handle_frontend_test(
    fn_tree="torch.can_cast",
    from_=helpers.get_dtypes("valid", full=False),
    to=helpers.get_dtypes("valid", full=False),
    test_with_out=st.just(False),
    number_positional_args=st.just(2),
)
# there are 100 combinations of dtypes, so run 200 examples to make sure all are tested
@settings(max_examples=200)
def test_torch_can_cast(
    *,
    from_,
    to,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    helpers.test_frontend_function(
        input_dtypes=[],
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        from_=ivy.Dtype(from_[0]),
        to=ivy.Dtype(to[0]),
    )


# promote_types
@handle_frontend_test(
    fn_tree="torch.promote_types",
    type1=helpers.get_dtypes("valid", full=False),
    type2=helpers.get_dtypes("valid", full=False),
    test_with_out=st.just(False),
)
# there are 100 combinations of dtypes, so run 200 examples to make sure all are tested
@settings(max_examples=200)
def test_torch_promote_types(
    *,
    type1,
    type2,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    ret, frontend_ret = helpers.test_frontend_function(
        input_dtypes=[],
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        type1=type1[0],
        type2=type2[0],
    )
    assert ret == repr(frontend_ret[0]).split(".")[1]


# set_default_dtype
@handle_frontend_test(
    fn_tree="torch.set_default_dtype",
    dtype=helpers.get_dtypes("float", full=False),
)
def test_set_default_dtype(
    *,
    dtype,
):
    dtype = dtype[0]
    torch_frontend.set_default_dtype(dtype)
    assert torch_frontend.get_default_dtype() == dtype
