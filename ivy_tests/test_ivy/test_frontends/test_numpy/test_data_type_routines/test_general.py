# global
from hypothesis import strategies as st, settings

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers.testing_helpers import handle_frontend_test


# can_cast
@handle_frontend_test(
    fn_tree="numpy.can_cast",
    from_=helpers.get_dtypes("valid", full=False),
    to=helpers.get_dtypes("valid", full=False),
    casting=st.sampled_from(["no", "equiv", "safe", "same_kind", "unsafe"]),
)
@settings(max_examples=200)
def test_numpy_can_cast(
    *,
    from_,
    to,
    casting,
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
        num_positional_args=3,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        from_=from_[0],
        to=to[0],
        casting=casting,
    )


# promote_types
@handle_frontend_test(
    fn_tree="numpy.promote_types",
    type1=helpers.get_dtypes("valid", full=False),
    type2=helpers.get_dtypes("valid", full=False),
)
# there are 100 combinations of dtypes, so run 200 examples to make sure all are tested
@settings(max_examples=200)
def test_numpy_promote_types(
    *,
    type1,
    type2,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    ret, frontend_ret = helpers.test_frontend_function(
        input_dtypes=[],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        type1=type1[0],
        type2=type2[0],
        test_values=False,
    )
    assert ret._ivy_dtype == frontend_ret[0].name
