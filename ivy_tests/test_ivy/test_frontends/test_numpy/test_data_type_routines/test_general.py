# global
from hypothesis import strategies as st, settings, assume

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers.testing_helpers import handle_frontend_test


# can_cast
@handle_frontend_test(
    fn_tree="numpy.can_cast",
    from_=helpers.get_dtypes("valid", full=False),
    to=helpers.get_dtypes("valid", full=False),
    casting=st.sampled_from(["no", "equiv", "safe", "same_kind", "unsafe"]),
    test_with_out=st.just(False),
    number_positional_args=st.just(3),
)
@settings(max_examples=200)
def test_numpy_can_cast(
    *,
    from_,
    to,
    casting,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    helpers.test_frontend_function(
        input_dtypes=[],
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        from_=from_[0],
        to=to[0],
        casting=casting,
    )


@handle_frontend_test(
    fn_tree="numpy.min_scalar_type",
    x=st.one_of(
        helpers.ints(min_value=-256, max_value=256),
        st.booleans(),
        helpers.floats(min_value=-256, max_value=256),
    ),
)
@settings(max_examples=200)
def test_numpy_min_scalar_type(
    *,
    x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):  # skip torch backend uint
    if ivy.current_backend_str() == "torch":
        assume(not isinstance(x, int))
    ret, frontend_ret = helpers.test_frontend_function(
        input_dtypes=[],
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x,
        test_values=False,
    )
    assert ret._ivy_dtype == frontend_ret[0].name


# promote_types
@handle_frontend_test(
    fn_tree="numpy.promote_types",
    type1=helpers.get_dtypes("valid", full=False),
    type2=helpers.get_dtypes("valid", full=False),
    test_with_out=st.just(False),
)
# there are 100 combinations of dtypes, so run 200 examples to make sure all are tested
@settings(max_examples=200)
def test_numpy_promote_types(
    *,
    type1,
    type2,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    ret, frontend_ret = helpers.test_frontend_function(
        input_dtypes=[],
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        type1=type1[0],
        type2=type2[0],
        test_values=False,
    )
    assert ret._ivy_dtype == frontend_ret[0].name
