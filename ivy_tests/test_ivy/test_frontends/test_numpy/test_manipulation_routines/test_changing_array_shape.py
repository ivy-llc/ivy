# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
from ivy_tests.test_ivy.helpers import handle_cmd_line_args

@st.composite
def _copy_n_with_out(draw):
    copy = draw(st.booleans())
    out = draw(st.booleans())
    return copy, out

@st.composite
def _x_dtypes_n_shape(draw):
    min_num_dims = 1
    max_num_dims = 5
    min_dim_size = 1
    max_dim_size = 10

    shape = []
    shape_len = draw(st.integers(min_num_dims, max_num_dims))
    for _ in range(shape_len):
        s_ = draw(st.integers(min_dim_size, max_dim_size))
        shape.append(s_)

    dtypes, x = draw( helpers.dtype_and_values(shape=tuple(shape)) )
    reshape = draw(st.permutations(shape))
    return dtypes, x, tuple(reshape)

# reshape
@handle_cmd_line_args
@given(
    x_dtypes_n_shape=_x_dtypes_n_shape(),
    copy_n_with_out=_copy_n_with_out(),
    as_variable=helpers.array_bools(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.reshape"
    ),
    native_array=helpers.array_bools(),
)
def test_numpy_reshape(
    x_dtypes_n_shape,
    copy_n_with_out,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    copy, with_out = copy_n_with_out
    dtypes, x, shape = x_dtypes_n_shape
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="reshape",
        x=x,
        shape=shape,
    )
