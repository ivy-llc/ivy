# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


@handle_frontend_test(
    fn_tree="jax.lax.cond",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        min_dim_size=1,
    ),
    pred_cond=st.booleans(),
    test_with_out=st.just(False),
)
def test_jax_cond(
    *,
    dtype_and_x,
    pred_cond,
    test_flags,
    on_device,
    fn_tree,
    frontend,
):
    def _test_true_fn(x):
        return x + x

    def _test_false_fn(x):
        return x * x

    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        pred=pred_cond,
        true_fun=_test_true_fn,
        false_fun=_test_false_fn,
        operand=x[0],
    )


@handle_frontend_test(
    fn_tree="jax.lax.map",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        min_dim_size=1,
    ),
    test_with_out=st.just(False),
)
def test_jax_map(
    *,
    dtype_and_x,
    test_flags,
    on_device,
    fn_tree,
    frontend,
):
    def _test_map_fn(x):
        return x + x

    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        f=_test_map_fn,
        xs=x[0],
    )


@handle_frontend_test(
    fn_tree="jax.lax.switch",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        min_dim_size=1,
    ),
    index=helpers.ints(min_value=-10, max_value=10),
    test_with_out=st.just(False),
)
def test_jax_switch(
    *,
    dtype_and_x,
    index,
    test_flags,
    on_device,
    fn_tree,
    frontend,
):
    def _test_branch_1(x):
        return x + x

    def _test_branch_2(x):
        return x * x

    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        index=index,
        branches=[_test_branch_1, _test_branch_2],
        operand=x[0],
    )
