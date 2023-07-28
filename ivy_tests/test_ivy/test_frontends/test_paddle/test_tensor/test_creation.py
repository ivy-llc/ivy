# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy_tests.test_ivy.helpers.globals as test_globals
from ivy_tests.test_ivy.helpers import handle_frontend_test, update_backend


# Helpers #
# ------- #


@st.composite
def _input_fill_and_dtype(draw):
    dtype = draw(helpers.get_dtypes("float", full=False))
    with update_backend(test_globals.CURRENT_BACKEND) as ivy_backend:
        dtype_and_input = draw(helpers.dtype_and_values(dtype=dtype))
        if ivy_backend.is_uint_dtype(dtype[0]):
            fill_values = draw(st.integers(min_value=0, max_value=5))
        elif ivy_backend.is_int_dtype(dtype[0]):
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
    backend_fw,
):
    input_dtype, input = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
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
    backend_fw,
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
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
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
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
    backend_fw,
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
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
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
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
    backend_fw,
    on_device,
):
    input_dtype, x, fill, dtype_to_cast = input_fill_dtype
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
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
    backend_fw,
    on_device,
):
    input_dtype, x, fill, dtype_to_cast = input_fill_dtype
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        fill_value=fill,
        dtype=dtype_to_cast,
    )


# arange
@handle_frontend_test(
    fn_tree="paddle.arange",
    start=helpers.ints(min_value=-50, max_value=0),
    end=helpers.ints(min_value=1, max_value=50),
    step=helpers.ints(min_value=1, max_value=5),
    dtype=helpers.get_dtypes("float"),
    test_with_out=st.just(False),
)
def test_paddle_arange(
    start,
    end,
    step,
    dtype,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        start=start,
        end=end,
        step=step,
        dtype=dtype[0],
    )


# empty
@handle_frontend_test(
    fn_tree="paddle.empty",
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtype=helpers.get_dtypes("valid", full=False),
    test_with_out=st.just(False),
)
def test_paddle_empty(
    shape,
    dtype,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        shape=shape,
        dtype=dtype[0],
    )


# eye
@handle_frontend_test(
    fn_tree="paddle.eye",
    num_rows=helpers.ints(min_value=3, max_value=10),
    num_columns=st.none() | helpers.ints(min_value=3, max_value=10),
    dtypes=helpers.get_dtypes("valid", full=False),
    test_with_out=st.just(False),
)
def test_paddle_eye(
    *,
    num_rows,
    num_columns,
    dtypes,
    on_device,
    fn_tree,
    test_flags,
    frontend,
    backend_fw,
):
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        num_rows=num_rows,
        num_columns=num_columns,
        dtype=dtypes[0],
    )


# empty_like
@handle_frontend_test(
    fn_tree="paddle.empty_like",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
    dtype=helpers.get_dtypes("valid", full=False),
    test_with_out=st.just(False),
)
def test_paddle_empty_like(
    dtype_and_x,
    dtype,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        x=x[0],
        dtype=dtype[0],
    )


# tril
@handle_frontend_test(
    fn_tree="paddle.tril",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,
    ),
    diagonal=st.integers(min_value=-100, max_value=100),
)
def test_paddle_tril(
    *,
    dtype_and_values,
    diagonal,
    backend_fw,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    dtype, values = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=values[0],
        diagonal=diagonal,
    )


# triu
@handle_frontend_test(
    fn_tree="paddle.triu",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,
    ),
    diagonal=st.integers(min_value=-100, max_value=100),
)
def test_paddle_triu(
    *,
    dtype_and_values,
    diagonal,
    on_device,
    backend_fw,
    fn_tree,
    frontend,
    test_flags,
):
    dtype, values = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=values[0],
        diagonal=diagonal,
    )


# diagflat
@handle_frontend_test(
    fn_tree="paddle.diagflat",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    offset=st.integers(min_value=-4, max_value=4),
    test_with_out=st.just(False),
)
def test_paddle_diagflat(
    dtype_and_values,
    offset,
    test_flags,
    backend_fw,
    frontend,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        x=x[0],
        offset=offset,
    )


@handle_frontend_test(
    fn_tree="paddle.meshgrid",
    dtype_and_arrays=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=st.integers(min_value=2, max_value=5),
        min_num_dims=1,
        max_num_dims=1,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_paddle_meshgrid(
    dtype_and_arrays,
    test_flags,
    backend_fw,
    frontend,
    fn_tree,
    on_device,
):
    input_dtype, arrays = dtype_and_arrays
    args = {}
    i = 0
    for x_ in arrays:
        args["x{}".format(i)] = x_
        i += 1
    test_flags.num_positional_args = len(arrays)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        **args,
    )


# assign
@handle_frontend_test(
    fn_tree="paddle.assign",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(True),
)
def test_paddle_assign(
    dtype_and_x,
    test_flags,
    backend_fw,
    frontend,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        output=x[1],
    )


# diag
@handle_frontend_test(
    fn_tree="paddle.diag",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        max_num_dims=2,
        min_dim_size=1,
        max_dim_size=5,
    ),
    k=helpers.ints(min_value=-1, max_value=1),
    p=st.one_of(
        helpers.ints(min_value=-25, max_value=25),
        helpers.floats(min_value=-25, max_value=25),
    ),
)
def test_paddle_diag(
    dtype_and_x,
    k,
    p,
    backend_fw,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        offset=k,
        padding_value=p,
    )


# logspace
@handle_frontend_test(
    fn_tree="paddle.logspace",
    start=helpers.floats(min_value=-10, max_value=10),
    stop=helpers.floats(min_value=-10, max_value=10),
    num=helpers.ints(min_value=1, max_value=5),
    base=st.floats(min_value=0.1, max_value=10.0),
    dtype=helpers.get_dtypes("float"),
    test_with_out=st.just(False),
)
def test_paddle_logspace(
    start,
    stop,
    num,
    base,
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
        start=start,
        stop=stop,
        num=num,
        base=base,
        dtype=dtype[0],
    )
