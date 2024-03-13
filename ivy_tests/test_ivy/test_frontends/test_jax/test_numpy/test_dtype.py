# global
from hypothesis import strategies as st, settings

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers.testing_helpers import handle_frontend_test
from ivy_tests.test_ivy.test_functional.test_core.test_statistical import (
    _get_castable_dtype,
)


@handle_frontend_test(
    fn_tree="jax.numpy.astype",
    dtype_and_x=_get_castable_dtype(),
    test_with_out=st.just(False),
)
def test_jax_astype(
    dtype_and_x,
    fn_tree,
    test_flags,
    on_device,
    frontend,
    backend_fw,
):
    input_dtype, x, _, castable_dtype = dtype_and_x

    helpers.test_frontend_function(
        input_dtypes=[input_dtype],
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x,
        dtype=castable_dtype,
    )


# can_cast
@handle_frontend_test(
    fn_tree="jax.numpy.can_cast",
    from_=helpers.get_dtypes("valid", full=False),
    to=helpers.get_dtypes("valid", full=False),
    casting=st.sampled_from(["no", "equiv", "safe", "same_kind", "unsafe"]),
    test_with_out=st.just(False),
)
# there are 100 combinations of dtypes, so run 200 examples to make sure all are tested
@settings(max_examples=200)
def test_jax_can_cast(
    *,
    from_,
    to,
    casting,
    test_flags,
    on_device,
    fn_tree,
    frontend,
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
    fn_tree="jax.numpy.finfo",
    dtype=helpers.get_dtypes("numeric", full=False),
    test_with_out=st.just(False),
)
def test_jax_finfo(*, dtype, test_flags, on_device, fn_tree, frontend, backend_fw):
    helpers.test_frontend_function(
        input_dtypes=[],
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        dtype=dtype[0],
        backend_to_test=backend_fw,
    )


@handle_frontend_test(
    fn_tree="jax.numpy.iinfo",
    dtype=helpers.get_dtypes("numeric", full=False),
    test_with_out=st.just(False),
)
@settings(max_examples=200)
def test_jax_iinfo(*, dtype, test_flags, on_device, fn_tree, frontend, backend_fw):
    helpers.test_frontend_function(
        input_dtypes=[],
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        int_type=dtype[0],
        backend_to_test=backend_fw,
    )


# promote_types
@handle_frontend_test(
    fn_tree="jax.numpy.promote_types",
    type1=helpers.get_dtypes("valid", full=False),
    type2=helpers.get_dtypes("valid", full=False),
    test_with_out=st.just(False),
)
# there are 100 combinations of dtypes, so run 200 examples to make sure all are tested
@settings(max_examples=200)
def test_jax_promote_types(
    *,
    type1,
    type2,
    test_flags,
    on_device,
    fn_tree,
    frontend,
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
    assert str(ret._ivy_dtype) == str(frontend_ret[0])


@handle_frontend_test(
    fn_tree="jax.numpy.result_type",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=st.shared(helpers.ints(min_value=2, max_value=5), key="num_arrays"),
        shared_dtype=False,
    ),
    test_with_out=st.just(False),
)
@settings(max_examples=200)
def test_jax_result_type(
    *, dtype_and_x, test_flags, on_device, fn_tree, frontend, backend_fw
):
    dtype, x = helpers.as_lists(*dtype_and_x)
    kw = {}
    for i, (dtype_, x_) in enumerate(zip(dtype, x)):
        kw[f"x{i}"] = x_
    test_flags.num_positional_args = len(kw)
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        **kw,
    )
