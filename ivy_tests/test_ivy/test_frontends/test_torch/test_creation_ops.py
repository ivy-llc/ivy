# global
import ivy
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# Helper functions
@st.composite
def _dtypes(draw):
    return draw(
        st.shared(
            helpers.list_of_length(
                x=st.sampled_from(draw(helpers.get_dtypes("numeric"))), length=1
            ),
            key="dtype",
        )
    )


@st.composite
def _fill_value(draw):
    dtype = draw(_dtypes())[0]
    if ivy.is_uint_dtype(dtype):
        return draw(helpers.ints(min_value=0, max_value=5))
    elif ivy.is_int_dtype(dtype):
        return draw(helpers.ints(min_value=-5, max_value=5))
    return draw(helpers.floats(min_value=-5, max_value=5))


@st.composite
def _requires_grad(draw):
    dtype = draw(_dtypes())[0]
    if ivy.is_int_dtype(dtype) or ivy.is_uint_dtype(dtype):
        return draw(st.just(False))
    return draw(st.booleans())


# full
@handle_cmd_line_args
@given(
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    fill_value=_fill_value(),
    dtypes=_dtypes(),
    requires_grad=_requires_grad(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.full"
    ),
)
def test_torch_full(
    shape,
    fill_value,
    dtypes,
    requires_grad,
    device,
    num_positional_args,
):
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        as_variable_flags=[False],
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=[False],
        device=device,
        frontend="torch",
        fn_tree="full",
        size=shape,
        fill_value=fill_value,
        dtype=dtypes[0],
        requires_grad=requires_grad,
    )


# ones_like
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    dtypes=_dtypes(),
    requires_grad=_requires_grad(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.ones_like"
    ),
)
def test_torch_ones_like(
    dtype_and_x,
    dtypes,
    requires_grad,
    device,
    num_positional_args,
):
    dtype, input = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=[False],
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=[False],
        device=device,
        frontend="torch",
        fn_tree="ones_like",
        input=input[0],
        dtype=dtypes[0],
        requires_grad=requires_grad,
    )


# ones
@handle_cmd_line_args
@given(
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtypes=_dtypes(),
    requires_grad=_requires_grad(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.ones"
    ),
)
def test_torch_ones(
    shape,
    dtypes,
    requires_grad,
    device,
    num_positional_args,
):
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        as_variable_flags=[False],
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=[False],
        device=device,
        frontend="torch",
        fn_tree="ones",
        size=shape,
        dtype=dtypes[0],
        requires_grad=requires_grad,
    )


# zeros
@handle_cmd_line_args
@given(
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtypes=_dtypes(),
    requires_grad=_requires_grad(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.zeros"
    ),
)
def test_torch_zeros(
    shape,
    dtypes,
    requires_grad,
    device,
    num_positional_args,
):
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        as_variable_flags=[False],
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=[False],
        device=device,
        frontend="torch",
        fn_tree="zeros",
        size=shape,
        dtype=dtypes[0],
        requires_grad=requires_grad,
    )


@handle_cmd_line_args
@given(
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtypes=helpers.get_dtypes("valid", full=False),
    requires_grad=_requires_grad(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.empty"
    ),
)
def test_torch_empty(
    shape,
    dtypes,
    requires_grad,
    device,
    num_positional_args,
):
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        as_variable_flags=[False],
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=[False],
        device=device,
        frontend="torch",
        fn_tree="empty",
        size=shape,
        dtype=dtypes,
        requires_grad=requires_grad,
    )


# arange


@handle_cmd_line_args
@given(
    start=helpers.ints(min_value=0, max_value=50),
    stop=helpers.ints(min_value=0, max_value=50),
    step=helpers.ints(min_value=-50, max_value=50).filter(
        lambda x: True if x != 0 else False
    ),
    dtypes=helpers.get_dtypes("float", full=False),
    requires_grad=_requires_grad(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.arange"
    ),
)
def test_torch_arange(
    start,
    stop,
    step,
    dtypes,
    requires_grad,
    device,
    num_positional_args,
):
    helpers.test_frontend_function(
        input_dtypes=[],
        as_variable_flags=[False],
        with_out=False,
        num_positional_args=3,
        native_array_flags=[False],
        device=device,
        frontend="torch",
        fn_tree="arange",
        end=stop,
        start=start,
        step=step,
        dtype=dtypes,
        requires_grad=requires_grad,
    )


# range


@handle_cmd_line_args
@given(
    start=helpers.ints(min_value=0, max_value=50),
    stop=helpers.ints(min_value=0, max_value=50),
    step=helpers.ints(min_value=-50, max_value=50).filter(
        lambda x: True if x != 0 else False
    ),
    dtypes=helpers.get_dtypes("float", full=False),
    requires_grad=_requires_grad(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.range"
    ),
)
def test_torch_range(
    start,
    stop,
    step,
    dtypes,
    requires_grad,
    device,
    num_positional_args,
):
    helpers.test_frontend_function(
        input_dtypes=[],
        as_variable_flags=[False],
        with_out=False,
        num_positional_args=3,
        native_array_flags=[False],
        device=device,
        frontend="torch",
        fn_tree="range",
        end=stop,
        start=start,
        step=step,
        dtype=dtypes,
        requires_grad=requires_grad,
    )


# linspace


@handle_cmd_line_args
@given(
    start=st.floats(min_value=-10, max_value=10),
    stop=st.floats(min_value=-10, max_value=10),
    num=st.integers(min_value=1, max_value=10),
    requires_grad=_requires_grad(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.linspace"
    ),
)
def test_torch_linspace(
    start,
    stop,
    num,
    requires_grad,
    device,
    num_positional_args,
):
    helpers.test_frontend_function(
        input_dtypes=[],
        as_variable_flags=[False],
        with_out=False,
        num_positional_args=3,
        native_array_flags=[False],
        device=device,
        frontend="torch",
        fn_tree="linspace",
        start=start,
        end=stop,
        num=num,
        requires_grad=requires_grad,
    )


# logspace


@handle_cmd_line_args
@given(
    start=st.floats(min_value=-10, max_value=10),
    stop=st.floats(min_value=-10, max_value=10),
    num=st.integers(min_value=1, max_value=10),
    requires_grad=_requires_grad(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.logspace"
    ),
)
def test_torch_logspace(
    start,
    stop,
    num,
    requires_grad,
    device,
    num_positional_args,
):
    helpers.test_frontend_function(
        input_dtypes=[],
        as_variable_flags=[False],
        with_out=False,
        num_positional_args=3,
        native_array_flags=[False],
        device=device,
        frontend="torch",
        fn_tree="logspace",
        start=start,
        end=stop,
        num=num,
        requires_grad=requires_grad,
    )


# empty_like
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    requires_grad=_requires_grad(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.empty_like"
    ),
)
def test_torch_empty_like(
    dtype_and_x,
    requires_grad,
    device,
    num_positional_args,
):
    dtype, input = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=[False],
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=[False],
        device=device,
        frontend="torch",
        fn_tree="empty_like",
        input=input[0],
        dtype=dtype,
        requires_grad=requires_grad,
    )


# full_like
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    fill_value=_fill_value(),
    dtypes=_dtypes(),
    requires_grad=_requires_grad(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.full_like"
    ),
)
def test_torch_full_like(
    dtype_and_x,
    fill_value,
    dtypes,
    requires_grad,
    device,
    num_positional_args,
):
    dtype, input = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=[False],
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=[False],
        device=device,
        frontend="torch",
        fn_tree="full_like",
        input=input[0],
        fill_value=fill_value,
        dtype=dtypes[0],
        requires_grad=requires_grad,
    )


# as_tensor and tensor by proxy
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
)
def test_torch_as_tensor(dtype_and_x):
    dtype, input = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=[False],
        with_out=False,
        num_positional_args=1,
        native_array_flags=[False],
        device="cpu",
        frontend="torch",
        fn_tree="as_tensor",
        input=input[0],
    )
