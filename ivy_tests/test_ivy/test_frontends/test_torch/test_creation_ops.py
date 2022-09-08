# global
import ivy
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# full
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
        fn_tree="full",
        size=shape,
        fill_value=fill_value,
        dtype=dtypes[0],
        device=device,
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
    fw,
):
    dtype, input = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=False,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=False,
        fw=fw,
        frontend="torch",
        fn_tree="ones_like",
        input=np.asarray(input, dtype=dtype),
        dtype=dtypes[0],
        device=device,
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
        fn_tree="ones",
        size=shape,
        dtype=dtypes[0],
        device=device,
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
        fn_tree="zeros",
        size=shape,
        dtype=dtypes[0],
        device=device,
        requires_grad=requires_grad,
    )
