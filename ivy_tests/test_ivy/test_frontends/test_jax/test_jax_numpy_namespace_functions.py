# global
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# abs
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("signed_integer"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.numpy.abs"
    ),
)
def test_jax_numpy_abs(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="numpy.abs",
        x=x[0],
    )


# allclose
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.numpy.allclose"
    ),
    equal_nan=st.booleans(),
)
def test_jax_numpy_allclose(
    dtype_and_input,
    equal_nan,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="numpy.allclose",
        a=input[0],
        b=input[1],
        rtol=1e-05,
        atol=1e-08,
        equal_nan=equal_nan,
    )


@st.composite
def _get_clip_inputs(draw):
    shape = draw(
        helpers.get_shape(
            min_num_dims=1, max_num_dims=5, min_dim_size=2, max_dim_size=10
        )
    )
    x_dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"),
            shape=shape,
        )
    )
    min = draw(st.booleans())
    if min:
        max = draw(st.booleans())
        min = draw(
            helpers.array_values(
                dtype=x_dtype[0], shape=shape, min_value=-50, max_value=5
            )
        )
        max = (
            draw(
                helpers.array_values(
                    dtype=x_dtype[0], shape=shape, min_value=5, max_value=50
                )
            )
            if max
            else None
        )
    else:
        min = None
        max = draw(
            helpers.array_values(
                dtype=x_dtype[0], shape=shape, min_value=0, max_value=50
            )
        )
    return x_dtype, x, min, max


# clip
@handle_cmd_line_args
@given(
    input_and_ranges=_get_clip_inputs(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.numpy.clip"
    ),
)
def test_jax_clip(
    input_and_ranges,
    num_positional_args,
    with_out,
    as_variable,
    native_array,
    fw,
):
    x_dtype, x, min, max = input_and_ranges
    helpers.test_frontend_function(
        input_dtypes=x_dtype,
        as_variable_flags=as_variable,
        num_positional_args=num_positional_args,
        with_out=with_out,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="numpy.clip",
        a=x[0],
        a_min=min,
        a_max=max,
        out=None,
    )
