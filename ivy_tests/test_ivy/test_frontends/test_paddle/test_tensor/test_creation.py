# global
import ivy
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# Helpers #
# ------- #


@st.composite
def _input_fill_and_dtype(draw):
    dtype = draw(helpers.get_dtypes("float", full=False))
    dtype_and_input = draw(helpers.dtype_and_values(dtype=dtype))
    if ivy.is_uint_dtype(dtype[0]):
        fill_values = draw(st.integers(min_value=0, max_value=5))
    elif ivy.is_int_dtype(dtype[0]):
        fill_values = draw(st.integers(min_value=-5, max_value=5))
    else:
        fill_values = draw(st.floats(min_value=-5, max_value=5))
    dtype_to_cast = draw(helpers.get_dtypes("float", full=False))
    return dtype, dtype_and_input[1], fill_values, dtype_to_cast[0]


# Tests #
# ----- #


# to_tensor
@handle_frontend_test(
    fn_tree="paddle.to_tensor",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
    dtype=helpers.get_dtypes("valid"),
)
def test_paddle_to_tensor(
    *,
    dtype_and_x,
    dtype,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, input = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        data=input[0],
        dtype=dtype[0],
        place=on_device,
    )


# ones
@handle_frontend_test(
    fn_tree="paddle.ones",
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtype=helpers.get_dtypes("valid"),
    test_with_out=st.just(False),
)
def test_paddle_ones(
    shape,
    dtype,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        shape=shape,
        dtype=dtype[0],
    )


# ones_like
@handle_frontend_test(
    fn_tree="paddle.ones_like",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    dtype=helpers.get_dtypes("valid"),
    test_with_out=st.just(False),
)
def test_paddle_ones_like(
    dtype_and_x,
    dtype,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        dtype=dtype[0],
    )


# zeros
@handle_frontend_test(
    fn_tree="paddle.zeros",
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtype=helpers.get_dtypes("valid"),
    test_with_out=st.just(False),
)
def test_paddle_zeros(
    shape,
    dtype,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        shape=shape,
        dtype=dtype[0],
    )


# zeros_like
@handle_frontend_test(
    fn_tree="paddle.zeros_like",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    dtype=helpers.get_dtypes("valid"),
    test_with_out=st.just(False),
)
def test_paddle_zeros_like(
    dtype_and_x,
    dtype,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        dtype=dtype[0],
    )


# full
@handle_frontend_test(
    fn_tree="paddle.full",
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    input_fill_dtype=_input_fill_and_dtype(),
    test_with_out=st.just(False),
)
def test_paddle_full(
    shape,
    input_fill_dtype,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x, fill, dtype_to_cast = input_fill_dtype
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        shape=shape,
        fill_value=fill,
        dtype=dtype_to_cast,
    )


# full_like
@handle_frontend_test(
    fn_tree="paddle.full_like",
    input_fill_dtype=_input_fill_and_dtype(),
    test_with_out=st.just(False),
)
def test_paddle_full_like(
    input_fill_dtype,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x, fill, dtype_to_cast = input_fill_dtype
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        fill_value=fill,
        dtype=dtype_to_cast,
    )
