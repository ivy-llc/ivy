# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
import ivy.functional.backends.torch as ivy_torch


@st.composite
def _dtype_x_shape(draw):
    return draw(
            helpers.dtype_and_values(
                available_dtypes=tuple(
                    set(ivy_np.valid_float_dtypes).intersection(
                    set(ivy_torch.valid_float_dtypes))),
                ret_shape=True), 
            )


@st.composite
def _integers(draw):
    return draw(st.integers(min_value=-num_dims, max_value=num_dims-1))


@st.composite
def _lists(draw):
    return draw(st.lists(elements=_integers(), min_size=1, max_size=num_dims, unique=True))


@st.composite
def _tuples(draw):
    return tuple(draw(_lists()))


# flip
@given(
    dtype_value_shape=_dtype_x_shape(),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="functional.frontends.torch.flip"
    ),
    native_array=st.booleans(),
    dims=st.one_of(_integers(), _lists(), _tuples()),
)
def test_torch_flip(
    dtype_value_shape,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
    dims,
):
    input_dtype, value, shape = dtype_value_shape
    global num_dims
    num_dims = len(shape)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_name="flip",
        input=np.asarray(value, dtype=input_dtype),
        dims=dims,
        out=None,
    )
