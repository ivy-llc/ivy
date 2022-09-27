# global
import ivy
import numpy as np
from hypothesis import given, assume, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# add
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.add"
    ),
)
def test_jax_lax_add(
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
        fn_tree="lax.add",
        x=x[0],
        y=x[1],
    )


# tan
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.tan"
    ),
)
def test_jax_lax_tan(
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
        fn_tree="lax.tan",
        x=x[0],
    )


# max
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.max"
    ),
)
def test_jax_lax_max(
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
        fn_tree="lax.max",
        x=x[0],
        y=x[1],
    )


# noinspection DuplicatedCode
@st.composite
def _arrays_idx_n_dtypes(draw):
    num_dims = draw(st.shared(helpers.ints(min_value=1, max_value=4), key="num_dims"))
    num_arrays = draw(
        st.shared(helpers.ints(min_value=2, max_value=4), key="num_arrays")
    )
    common_shape = draw(
        helpers.lists(
            arg=helpers.ints(min_value=2, max_value=4),
            min_size=num_dims - 1,
            max_size=num_dims - 1,
        )
    )
    unique_idx = draw(helpers.ints(min_value=0, max_value=num_dims - 1))
    unique_dims = draw(
        helpers.lists(
            arg=helpers.ints(min_value=2, max_value=3),
            min_size=num_arrays,
            max_size=num_arrays,
        )
    )
    xs = list()
    input_dtypes = draw(
        helpers.array_dtypes(
            available_dtypes=draw(helpers.get_dtypes("numeric")),
            shared_dtype=True,
        )
    )
    for ud, dt in zip(unique_dims, input_dtypes):
        x = draw(
            helpers.array_values(
                shape=common_shape[:unique_idx] + [ud] + common_shape[unique_idx:],
                dtype=dt,
            )
        )
        xs.append(x)
    return xs, input_dtypes, unique_idx


# concat
@handle_cmd_line_args
@given(
    xs_n_input_dtypes_n_unique_idx=_arrays_idx_n_dtypes(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.concatenate"
    ),
)
def test_jax_lax_concat(
    xs_n_input_dtypes_n_unique_idx,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    xs, input_dtypes, unique_idx = xs_n_input_dtypes_n_unique_idx
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="lax.concatenate",
        operands=xs,
        dimension=unique_idx,
    )


# full
@st.composite
def _dtypes(draw):
    return draw(
        st.shared(
            helpers.list_of_length(
                x=st.sampled_from(ivy.valid_numeric_dtypes), length=1
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
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.full"
    ),
)
def test_jax_lax_full(
    shape,
    fill_value,
    dtypes,
    num_positional_args,
    as_variable,
    native_array,
    fw,
):
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="lax.full",
        shape=shape,
        fill_value=fill_value,
        dtype=dtypes[0],
    )


# abs
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("signed_integer"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.abs"
    ),
)
def test_jax_lax_abs(
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
        fn_tree="lax.abs",
        x=x[0],
    )


# sqrt
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.sqrt"
    ),
)
def test_jax_lax_sqrt(
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
        fn_tree="lax.sqrt",
        x=x[0],
    )


# acos
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.acos"
    ),
)
def test_jax_lax_acos(
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
        fn_tree="lax.acos",
        x=x[0],
    )


# sin
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.sin"
    ),
)
def test_jax_lax_sin(
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
        fn_tree="lax.sin",
        x=x[0],
    )


# sign
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric")
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.sign"
    ),
)
def test_jax_lax_sign(
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
        fn_tree="lax.sign",
        x=x[0],
    )


# asin
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.asin"
    ),
)
def test_jax_lax_asin(
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
        fn_tree="lax.asin",
        x=x[0],
    )


# sinh
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.sinh"
    ),
)
def test_jax_lax_sinh(
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
        fn_tree="lax.sinh",
        x=x[0],
    )


# atan2
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.atan2"
    ),
)
def test_jax_lax_atan2(
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
        fn_tree="lax.atan2",
        x=x[0],
        y=x[1],
    )


@handle_cmd_line_args
@given(
    dtypes_and_xs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.min"
    ),
)
def test_jax_lax_min(
    dtypes_and_xs,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtypes, xs = dtypes_and_xs
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="lax.min",
        x=xs[0],
        y=xs[1],
    )


@handle_cmd_line_args
@given(
    dtypes_and_xs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.eq"
    ),
)
def test_jax_lax_eq(
    dtypes_and_xs,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtypes, xs = dtypes_and_xs
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="lax.eq",
        x=xs[0],
        y=xs[1],
    )


@handle_cmd_line_args
@given(
    dtypes_and_xs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.mul"
    ),
)
def test_jax_lax_mul(
    dtypes_and_xs,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtypes, xs = dtypes_and_xs
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="lax.mul",
        x=xs[0],
        y=xs[1],
    )


# atan
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.atan"
    ),
)
def test_jax_lax_atan(
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
        fn_tree="lax.atan",
        x=x[0],
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.ceil"
    ),
)
def test_jax_lax_ceil(
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
        fn_tree="lax.ceil",
        x=x[0],
    )


# bitwise_and
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.bitwise_and"
    ),
)
def test_jax_lax_bitwise_and(
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
        fn_tree="lax.bitwise_and",
        x=x[0],
        y=x[1],
    )


# bitwise_or
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.bitwise_or"
    ),
)
def test_jax_lax_bitwise_or(
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
        fn_tree="lax.bitwise_or",
        x=x[0],
        y=x[1],
    )


# bitwise_not
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=1,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.bitwise_not"
    ),
)
def test_jax_lax_bitwise_not(
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
        fn_tree="lax.bitwise_not",
        x=x[0],
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("signed_integer"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.neg"
    ),
)
def test_jax_lax_neg(
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
        fn_tree="lax.neg",
        x=x[0],
    )


@handle_cmd_line_args
@given(
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        min_dim_size=1,
        valid_axis=True,
        force_int_axis=True,
        allow_neg_axes=False,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.argmax"
    ),
    index_dtype=helpers.get_dtypes("integer", full=False),
)
def test_jax_lax_argmax(
    dtype_x_axis,
    as_variable,
    num_positional_args,
    native_array,
    fw,
    index_dtype,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="lax.argmax",
        operand=x[0],
        axis=axis,
        index_dtype=index_dtype,
    )


@handle_cmd_line_args
@given(
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        min_dim_size=1,
        valid_axis=True,
        force_int_axis=True,
        allow_neg_axes=False,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.argmin"
    ),
    index_dtype=helpers.get_dtypes("integer", full=False),
)
def test_jax_lax_argmin(
    dtype_x_axis,
    as_variable,
    num_positional_args,
    native_array,
    fw,
    index_dtype,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="lax.argmin",
        operand=x[0],
        axis=axis,
        index_dtype=index_dtype,
    )


# bitwise_xor
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.bitwise_xor"
    ),
)
def test_jax_lax_bitwise_xor(
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
        fn_tree="lax.bitwise_xor",
        x=x[0],
        y=x[1],
    )


@st.composite
def _dtype_and_values(draw, **kwargs):
    return draw(
        helpers.dtype_and_values(
            **kwargs,
            dtype=draw(_dtypes()),
        )
    )


@st.composite
def _shape_or_none(draw):
    return draw(helpers.get_shape() | st.none())


@handle_cmd_line_args
@given(
    dtype_and_x=_dtype_and_values(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.full_like"
    ),
    fill_val=_fill_value(),
    shape=_shape_or_none(),
)
def test_jax_lax_full_like(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    fw,
    fill_val,
    shape,
):
    dtype, x = dtype_and_x
    fill_val = fill_val
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="lax.full_like",
        x=x[0],
        fill_value=fill_val,
        dtype=dtype,
        shape=shape,
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.exp"
    ),
)
def test_jax_lax_exp(
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
        fn_tree="lax.exp",
        x=x[0],
    )


@st.composite
def _sample_castable_numeric_dtype(draw):
    dtype = draw(_dtypes())[0]
    to_dtype = draw(helpers.get_dtypes("numeric", full=False))
    assume(ivy.can_cast(dtype, to_dtype))
    return to_dtype


@handle_cmd_line_args
@given(
    dtype_and_x=_dtype_and_values(
        num_arrays=1,
        min_num_dims=1,
        min_value=-5,
        max_value=5,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.convert_element_type"
    ),
    new_dtype=_sample_castable_numeric_dtype(),
)
def test_jax_lax_convert_element_type(
    dtype_and_x, as_variable, num_positional_args, native_array, fw, new_dtype
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
        fn_tree="lax.convert_element_type",
        operand=x[0],
        new_dtype=new_dtype,
    )


@handle_cmd_line_args
@given(
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=5,
        valid_axis=True,
        allow_neg_axes=False,
        max_axes_size=1,
        force_int_axis=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.cumprod"
    ),
    reverse=st.booleans(),
)
def test_jax_lax_cumprod(
    dtype_x_axis,
    as_variable,
    num_positional_args,
    native_array,
    fw,
    reverse,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="lax.cumprod",
        operand=x[0],
        axis=axis,
        reverse=reverse,
    )


@handle_cmd_line_args
@given(
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=5,
        valid_axis=True,
        allow_neg_axes=False,
        max_axes_size=1,
        force_int_axis=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.cumsum"
    ),
    reverse=st.booleans(),
)
def test_jax_lax_cumsum(
    dtype_x_axis,
    as_variable,
    num_positional_args,
    native_array,
    fw,
    reverse,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="lax.cumsum",
        operand=x[0],
        axis=axis,
        reverse=reverse,
    )


@handle_cmd_line_args
@given(
    dtypes_and_xs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.ge"
    ),
)
def test_jax_lax_ge(
    dtypes_and_xs,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtypes, xs = dtypes_and_xs
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="lax.ge",
        x=xs[0],
        y=xs[1],
    )


@st.composite
def _reshape_helper(draw):
    # generate a shape s.t len(shape) > 0
    shape = draw(helpers.get_shape(min_num_dims=1))

    reshape_shape = draw(helpers.reshape_shapes(shape=shape))

    dtype = draw(helpers.array_dtypes(num_arrays=1))
    x = draw(helpers.array_values(dtype=dtype[0], shape=shape))

    is_dim = draw(st.booleans())
    if is_dim:
        # generate a permutation of [0, 1, 2, ... len(shape) - 1]
        permut = draw(st.permutations(list(range(len(shape)))))
        return x, dtype, reshape_shape, permut
    else:
        return x, dtype, reshape_shape, None


@handle_cmd_line_args
@given(
    x_reshape_permut=_reshape_helper(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.reshape"
    ),
)
def test_jax_lax_reshape(
    x_reshape_permut,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    x, dtype, shape, dimensions = x_reshape_permut
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="lax.reshape",
        operand=x[0],
        new_sizes=shape,
        dimensions=dimensions,
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.broadcast"
    ),
    sizes=helpers.get_shape(min_num_dims=1),
)
def test_jax_lax_broadcast(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    fw,
    sizes,
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
        fn_tree="lax.broadcast",
        operand=x[0],
        sizes=sizes,
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.reciprocal"
    ),
)
def test_jax_lax_reciprocal(
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
        fn_tree="lax.reciprocal",
        x=x[0],
    )


@handle_cmd_line_args
@given(
    dtype_x_bounded_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.sort"
    ),
    is_stable=st.booleans(),
)
def test_jax_lax_sort(
    dtype_x_bounded_axis,
    as_variable,
    num_positional_args,
    native_array,
    fw,
    is_stable,
):
    input_dtype, x, axis = dtype_x_bounded_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="lax.sort",
        operand=x[0],
        dimension=axis,
        is_stable=is_stable,
    )


@handle_cmd_line_args
@given(
    dtypes_and_xs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.le"
    ),
)
def test_jax_lax_le(
    dtypes_and_xs,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtypes, xs = dtypes_and_xs
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="lax.le",
        x=xs[0],
        y=xs[1],
    )


@handle_cmd_line_args
@given(
    dtypes_and_xs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.ne"
    ),
)
def test_jax_lax_ne(
    dtypes_and_xs,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtypes, xs = dtypes_and_xs
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="lax.ne",
        x=xs[0],
        y=xs[1],
    )


# cosh
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.cosh"
    ),
)
def test_jax_lax_cosh(
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
        fn_tree="lax.cosh",
        x=x[0],
    )


@handle_cmd_line_args
@given(
    dtypes_and_xs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.lt"
    ),
)
def test_jax_lax_lt(
    dtypes_and_xs,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtypes, xs = dtypes_and_xs
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="lax.lt",
        x=xs[0],
        y=xs[1],
    )


# round
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.round"
    ),
)
def test_jax_lax_round(
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
        fn_tree="lax.round",
        x=x[0],
    )


@handle_cmd_line_args
@given(
    dtypes_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.pow"
    ),
)
def test_jax_lax_pow(
    dtypes_and_values,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtypes, xs = dtypes_and_values
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="lax.pow",
        x=xs[0],
        y=xs[1],
    )


@handle_cmd_line_args
@given(
    dtypes_and_xs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.gt"
    ),
)
def test_jax_lax_gt(
    dtypes_and_xs,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtypes, xs = dtypes_and_xs
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="lax.gt",
        x=xs[0],
        y=xs[1],
    )


# cos
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.cos"
    ),
)
def test_jax_lax_cos(
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
        fn_tree="lax.cos",
        x=x[0],
    )


@st.composite
def _same_dims_min_x_max(draw):
    dtype = draw(helpers.get_dtypes("numeric", full=False))
    bound = draw(
        helpers.array_values(
            dtype=dtype,
            shape=(),
            large_abs_safety_factor=1.5,
            small_abs_safety_factor=1.5,
            safety_factor_scale="log",
        )
    )
    dtypes, x, shape = draw(
        helpers.dtype_and_values(
            dtype=[dtype],
            ret_shape=True,
        )
    )
    min_val = draw(
        helpers.array_values(
            dtype=dtype, shape=shape, max_value=bound, exclude_max=False
        )
    )
    max_val = draw(helpers.array_values(dtype=dtype, shape=shape, min_value=bound))
    return dtypes, x, min_val, max_val


@st.composite
def _basic_min_x_max(draw):
    dtype, value = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"),
        )
    )
    min_val = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=(),
            large_abs_safety_factor=1.5,
            small_abs_safety_factor=1.5,
            safety_factor_scale="log",
            exclude_max=False,
        )
    )
    max_val = draw(helpers.array_values(dtype=dtype[0], shape=(), min_value=min_val))
    return dtype, value, min_val, max_val


@handle_cmd_line_args
@given(
    dtype_x_min_max=(_same_dims_min_x_max() | _basic_min_x_max()),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.clamp"
    ),
)
def test_jax_lax_clamp(
    dtype_x_min_max,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtypes, x, min_vals, max_vals = dtype_x_min_max
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="lax.clamp",
        min=min_vals,
        x=x[0],
        max=max_vals,
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=1,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.log"
    ),
)
def test_jax_lax_log(
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
        fn_tree="lax.log",
        x=x[0],
    )


@handle_cmd_line_args
@given(
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        min_dim_size=2,
        valid_axis=True,
        force_int_axis=True,
        allow_neg_axes=False,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.rev"
    ),
)
def test_jax_lax_rev(
    dtype_x_axis,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="lax.rev",
        operand=x[0],
        dimensions=(axis,),
    )


@st.composite
def _div_dtypes_and_xs(draw):
    dtype, dividend, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"), ret_shape=True
        )
    )
    divisor = draw(
        helpers.array_values(dtype=dtype[0], min_value=1, max_value=20, shape=shape)
        | helpers.array_values(dtype=dtype[0], min_value=-20, max_value=-1, shape=shape)
    )
    return dtype, [dividend[0], divisor]


@handle_cmd_line_args
@given(
    dtypes_and_xs=_div_dtypes_and_xs(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.div"
    ),
)
def test_jax_lax_div(
    dtypes_and_xs,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtypes, xs = dtypes_and_xs
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="lax.div",
        x=xs[0],
        y=xs[1],
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.rsqrt"
    ),
)
def test_jax_lax_rsqrt(
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
        fn_tree="lax.rsqrt",
        x=x[0],
        rtol=1e-02,
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.expm1"
    ),
)
def test_jax_lax_expm1(
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
        fn_tree="lax.expm1",
        x=x[0],
    )


# log1p
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.log1p"
    ),
)
def test_jax_lax_log1p(
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
        fn_tree="lax.log1p",
        x=x[0],
    )


@st.composite
def _dtype_values_dims(draw):
    dtype, values, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"),
            min_num_dims=1,
            ret_shape=True,
        )
    )
    size = len(shape)
    permutations = draw(
        st.lists(
            st.integers(min_value=0, max_value=len(shape) - 1),
            min_size=size,
            max_size=size,
            unique=True,
        )
    )
    return dtype, values, tuple(permutations)


@handle_cmd_line_args
@given(
    dtype_x_dims=_dtype_values_dims(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.transpose"
    ),
)
def test_jax_lax_transpose(
    dtype_x_dims,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x, dims = dtype_x_dims
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="lax.transpose",
        operand=x[0],
        permutation=dims,
    )


@st.composite
def _two_valid_xs(draw):
    n = draw(st.integers(min_value=0, max_value=5))
    m = draw(st.integers(min_value=0, max_value=5))
    k = draw(st.integers(min_value=0, max_value=5))
    valid_shapes = [
        ((n, 1), (1, n)),
        ((n, 1), (n, 1)),
        ((m, k), (k, 1)),
        ((m, k), (k, n)),
    ]
    s1, s2 = draw(st.sampled_from(valid_shapes))
    d1, v1 = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            shape=s1,
        )
    )
    d2, v2 = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            shape=s2,
        )
    )
    return (d1, d2), (v1, v2)


@handle_cmd_line_args
@given(
    dtypes_and_xs=_two_valid_xs(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.dot"
    ),
    dtype=helpers.get_dtypes("valid", full=False, none=True),
)
def test_jax_lax_dot(
    dtypes_and_xs,
    as_variable,
    num_positional_args,
    native_array,
    fw,
    dtype,
):
    input_dtypes, xs = dtypes_and_xs
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="lax.dot",
        lhs=xs[0],
        rhs=xs[1],
        precision=None,
        preferred_element_type=dtype,
    )


@handle_cmd_line_args
@given(
    x_f_d_df=helpers.x_and_filters(dim=2),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.conv"
    ),
)
def test_jax_lax_conv(
    x_f_d_df,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    dtype, x, filters, dilations, data_format, stride, pad = x_f_d_df
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="lax.conv",
        lhs=x[0],
        rhs=filters[0],
        window_strides=(stride, stride),
        padding=pad,
    )


@handle_cmd_line_args
@given(
    x_f_d_df=helpers.x_and_filters(dim=2),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.conv_transpose"
    ),
)
def test_jax_lax_conv_transpose(
    x_f_d_df,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    dtype, x, filters, dilations, data_format, stride, pad = x_f_d_df
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="lax.conv_transpose",
        lhs=x[0],
        rhs=filters[0],
        strides=(stride, stride),
        padding=pad,
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.sub"
    ),
)
def test_jax_lax_sub(
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
        fn_tree="lax.sub",
        x=x[0],
        y=x[1],
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
        allow_inf=False,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.rem"
    ),
)
def test_jax_lax_rem(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x[0], 0)))  # ToDO, should use safety factor?
    assume(not np.any(np.isclose(x[1], 0)))
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="lax.rem",
        x=x[0],
        y=x[1],
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric")
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.square"
    ),
)
def test_jax_lax_square(
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
        fn_tree="lax.square",
        x=x[0],
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", full=True)
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.erf"
    ),
)
def test_jax_lax_erf(
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
        fn_tree="lax.erf",
        x=x[0],
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="functional.frontends.jax.lax.shift_left"
    ),
)
def test_jax_lax_shift_left(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x
    # negative shifts will throw an exception
    # shifts >= dtype witdth produce backend-defined behavior
    x[1] = np.clip(x[1], 0, np.iinfo(input_dtype[1]).bits - 1)

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="lax.shift_left",
        x=np.asarray(x[0], dtype=input_dtype[0]),
        y=np.asarray(x[1], dtype=input_dtype[1]),
    )
