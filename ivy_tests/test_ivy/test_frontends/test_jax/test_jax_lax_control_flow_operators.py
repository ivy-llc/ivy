# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test

# from typing import Callable, List, Tuple


@handle_frontend_test(
    fn_tree="jax.lax.cond",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        min_dim_size=1,
        large_abs_safety_factor=2,
        safety_factor_scale="log",
    ),
    pred_cond=st.booleans(),
    test_with_out=st.just(False),
)
def test_jax_lax_cond(
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


@handle_frontend_test(
    fn_tree="jax.lax.fori_loop",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_value=-1000,
        max_value=1000,
        min_num_dims=1,
        min_dim_size=1,
    ),
    lower=st.integers(min_value=-10, max_value=10),
    upper=st.integers(min_value=-10, max_value=10),
    test_with_out=st.just(False),
)
def test_jax_lax_fori_loop(
    *,
    dtype_and_x,
    lower,
    upper,
    test_flags,
    on_device,
    fn_tree,
    frontend,
):
    def _test_body_fn(x, y):
        return x + y

    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        lower=lower,
        upper=upper,
        body_fun=_test_body_fn,
        init_val=x[0],
    )


@handle_frontend_test(
    fn_tree="jax.lax.while_loop",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_value=-1000,
        max_value=1000,
        min_num_dims=1,
        min_dim_size=1,
    ),
)
def test_jax_lax_while_loop(
    *,
    dtype_and_x,
    test_flags,
    on_device,
    fn_tree,
    frontend,
):
    def _test_cond_fn(x):
        def any_negative_real(arr):
            for elem in arr:
                if isinstance(elem, (int, float)) and elem < 0:
                    return True
                elif isinstance(elem, complex):
                    return False
                elif isinstance(elem, (list, tuple)):
                    if any_negative_real(elem):
                        return True
            return False

        return any_negative_real(x)

    def _test_body_fn(x):
        return x + 1

    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        cond_fun=_test_cond_fn,
        body_fun=_test_body_fn,
        init_val=x[0],
    )


@handle_frontend_test(
    fn_tree="jax.lax.scan",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_value=-1000,
        max_value=1000,
        min_num_dims=1,
        min_dim_size=1,
    ),
)
def test_jax_lax_scan(
    *,
    dtype_and_x,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    unroll=1,
):
    def _test_f(carry, x):
        y = carry * x
        new_carry = x + y
        return new_carry, y

    input_dtype, x = dtype_and_x
    length = len(x)
    expected_carry = 0
    expected_ys = []
    carry = expected_carry
    for _ in range(unroll):
        for elem in x:
            carry, y = _test_f(carry, elem)
            expected_ys.append(y)

    # expected_result = (expected_carry, ivy.stack(expected_ys))

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        f=_test_f,
        init=expected_carry,
        xs=x,
        length=length,
        reverse=False,
        unroll=unroll,
        # expected_result=expected_result,
    )
