import hypothesis.extra.numpy as hnp
from hypothesis import given, strategies as st
import numpy as np

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@st.composite
def _broadcastable_trio(draw):
    dtype = draw(st.sampled_from(draw(helpers.get_dtypes("valid"))))

    shapes_st = hnp.mutually_broadcastable_shapes(num_shapes=3, min_dims=1, min_side=1)
    cond_shape, x1_shape, x2_shape = draw(shapes_st).input_shapes
    cond = draw(hnp.arrays(hnp.boolean_dtypes(), cond_shape))
    x1 = draw(hnp.arrays(dtype, x1_shape))
    x2 = draw(hnp.arrays(dtype, x2_shape))
    return cond, x1, x2, dtype


@handle_cmd_line_args
@given(
    broadcastables=_broadcastable_trio(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.where"
    ),
)
def test_numpy_where(
    broadcastables,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    cond, x1, x2, dtype = broadcastables

    helpers.test_frontend_function(
        input_dtypes=["bool", dtype, dtype],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="where",
        cond=cond,
        x1=x1,
        x2=x2,
    )


@handle_cmd_line_args
@given(
    dtype_and_a=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.nonzero"
    ),
)
def test_numpy_nonzero(
    dtype_and_a,
    native_array,
    num_positional_args,
    fw,
):
    dtype, a = dtype_and_a
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=False,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="nonzero",
        a=np.asarray(a, dtype=dtype),
    )


@st.composite
def _dtype_x_bounded_axis(draw, **kwargs):
    dtype, x, shape = draw(helpers.dtype_and_values(**kwargs, ret_shape=True))
    axis = draw(helpers.ints(min_value=0, max_value=len(shape) - 1))
    return dtype, x, axis


@handle_cmd_line_args
@given(
    dtype_x_axis=_dtype_x_bounded_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        min_dim_size=1,
    ),
    dtype=helpers.get_dtypes("float", full=False, none=True),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.argmin"
    ),
    keep_dims=st.booleans(),
)
def test_numpy_argmin(
    dtype_x_axis,
    dtype,
    as_variable,
    num_positional_args,
    native_array,
    fw,
    keep_dims,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="argmin",
        x=np.asarray(x, dtype=input_dtype),
        axis=axis,
        keepdims=keep_dims,
        out=None,
    )
