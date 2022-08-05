# global
import ivy
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.torch as ivy_torch


# flip
@st.composite
def _dtypes(draw):
    return draw(
        st.shared(
            helpers.list_of_length(
                x=st.sampled_from(ivy_torch.valid_numeric_dtypes), length=1
            ),
            key="dtype",
        )
    )

@st.composite
def _fill_value(draw):
    dtype = draw(_dtypes())[0]
    if ivy.is_uint_dtype(dtype):
        return draw(st.integers(0, 5))
    elif ivy.is_int_dtype(dtype):
        return draw(st.integers(-5, 5))
    return draw(st.floats(-5, 5))

@st.composite
def _requires_grad(draw):
    dtype = draw(_dtypes())[0]
    if ivy.is_int_dtype(dtype) or ivy.is_uint_dtype(dtype):
        return draw(st.just(False))
    return draw(st.booleans())


# flip
@given(
    shape=helpers.get_shape(
        allow_none=True,
        min_num_dims=0,
        max_num_dims=5,
        min_dim_size=0,
        max_dim_size=10,
    ),
    fill_value=_fill_value(),
    dtypes=_dtypes(),
    requires_grad=_requires_grad(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.flip"
    ),
)
def test_torch_flip(
    shape,
    fill_value,
    dtypes,
    requires_grad,
    device,
    num_positional_args,
    fw,
):
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        as_variable_flags=False,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=False,
        fw=fw,
        frontend="torch",
        fn_name="flip",
        size=shape,
        fill_value=fill_value,
        dtype=dtypes[0],
        device=device,
        requires_grad=requires_grad,
    )


# fliplr
@given(
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=0,
        max_dim_size=10,
    ),
    fill_value=_fill_value(),
    dtypes=_dtypes(),
    requires_grad=_requires_grad(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.fliplr"
    ),
)
def test_torch_fliplr(
    shape,
    fill_value,
    dtypes,
    requires_grad,
    device,
    num_positional_args,
    fw,
):
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        as_variable_flags=False,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=False,
        fw=fw,
        frontend="torch",
        fn_name="fliplr",
        size=shape,
        fill_value=fill_value,
        dtype=dtypes[0],
        device=device,
        requires_grad=requires_grad,
    )


# flipud
@given(
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=0,
        max_dim_size=10,
    ),
    fill_value=_fill_value(),
    dtypes=_dtypes(),
    requires_grad=_requires_grad(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.flipud"
    ),
)
def test_torch_flipud(
    shape,
    fill_value,
    dtypes,
    requires_grad,
    device,
    num_positional_args,
    fw,
):
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        as_variable_flags=False,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=False,
        fw=fw,
        frontend="torch",
        fn_name="flipud",
        size=shape,
        fill_value=fill_value,
        dtype=dtypes[0],
        device=device,
        requires_grad=requires_grad,
    )