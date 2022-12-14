# global
import ivy
import numpy as np
from hypothesis import assume, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# add
@handle_frontend_test(
    fn_tree="jax.lax.add",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_jax_lax_add(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        y=x[1],
    )


# tan
@handle_frontend_test(
    fn_tree="jax.lax.tan",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_jax_lax_tan(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# max
@handle_frontend_test(
    fn_tree="jax.lax.max",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_jax_lax_max(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
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
@handle_frontend_test(
    fn_tree="jax.lax.concatenate",
    xs_n_input_dtypes_n_unique_idx=_arrays_idx_n_dtypes(),
)
def test_jax_lax_concat(
    *,
    xs_n_input_dtypes_n_unique_idx,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    xs, input_dtypes, unique_idx = xs_n_input_dtypes_n_unique_idx
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        operands=xs,
        dimension=unique_idx,
    )


@st.composite
def _fill_value(draw):
    dtype = draw(helpers.get_dtypes("numeric", full=False, key="dtype"))[0]
    if ivy.is_uint_dtype(dtype):
        return draw(helpers.ints(min_value=0, max_value=5))
    elif ivy.is_int_dtype(dtype):
        return draw(helpers.ints(min_value=-5, max_value=5))
    return draw(helpers.floats(min_value=-5, max_value=5))


@handle_frontend_test(
    fn_tree="jax.lax.full",
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    fill_value=_fill_value(),
    dtypes=helpers.get_dtypes("numeric", full=False, key="dtype"),
)
def test_jax_lax_full(
    *,
    shape,
    fill_value,
    dtypes,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        shape=shape,
        fill_value=fill_value,
        dtype=dtypes[0],
    )


# abs
@handle_frontend_test(
    fn_tree="jax.lax.abs",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("signed_integer"),
    ),
)
def test_jax_lax_abs(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# sqrt
@handle_frontend_test(
    fn_tree="jax.lax.sqrt",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_jax_lax_sqrt(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# acos
@handle_frontend_test(
    fn_tree="jax.lax.acos",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_jax_lax_acos(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# sin
@handle_frontend_test(
    fn_tree="jax.lax.sin",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_jax_lax_sin(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# sign
@handle_frontend_test(
    fn_tree="jax.lax.sign",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric")
    ),
)
def test_jax_lax_sign(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# asin
@handle_frontend_test(
    fn_tree="jax.lax.asin",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_jax_lax_asin(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# sinh
@handle_frontend_test(
    fn_tree="jax.lax.sinh",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_jax_lax_sinh(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# atan2
@handle_frontend_test(
    fn_tree="jax.lax.atan2",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_jax_lax_atan2(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        y=x[1],
    )


@handle_frontend_test(
    fn_tree="jax.lax.min",
    dtypes_and_xs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_jax_lax_min(
    *,
    dtypes_and_xs,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtypes, xs = dtypes_and_xs
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xs[0],
        y=xs[1],
    )


@handle_frontend_test(
    fn_tree="jax.lax.eq",
    dtypes_and_xs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_jax_lax_eq(
    *,
    dtypes_and_xs,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtypes, xs = dtypes_and_xs
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xs[0],
        y=xs[1],
    )


@handle_frontend_test(
    fn_tree="jax.lax.mul",
    dtypes_and_xs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_jax_lax_mul(
    *,
    dtypes_and_xs,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtypes, xs = dtypes_and_xs
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xs[0],
        y=xs[1],
    )


# atan
@handle_frontend_test(
    fn_tree="jax.lax.atan",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_jax_lax_atan(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


@handle_frontend_test(
    fn_tree="jax.lax.ceil",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_jax_lax_ceil(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# bitwise_and
@handle_frontend_test(
    fn_tree="jax.lax.bitwise_and",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_jax_lax_bitwise_and(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        y=x[1],
    )


# bitwise_or
@handle_frontend_test(
    fn_tree="jax.lax.bitwise_or",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_jax_lax_bitwise_or(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        y=x[1],
    )


# bitwise_not
@handle_frontend_test(
    fn_tree="jax.lax.bitwise_not",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=1,
    ),
)
def test_jax_lax_bitwise_not(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


@handle_frontend_test(
    fn_tree="jax.lax.neg",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("signed_integer"),
    ),
)
def test_jax_lax_neg(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


@handle_frontend_test(
    fn_tree="jax.lax.argmax",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        min_dim_size=1,
        valid_axis=True,
        force_int_axis=True,
        allow_neg_axes=False,
    ),
    index_dtype=helpers.get_dtypes("integer", full=False),
)
def test_jax_lax_argmax(
    *,
    dtype_x_axis,
    index_dtype,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        operand=x[0],
        axis=axis,
        index_dtype=index_dtype[0],
    )


@handle_frontend_test(
    fn_tree="jax.lax.argmin",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        min_dim_size=1,
        valid_axis=True,
        force_int_axis=True,
        allow_neg_axes=False,
    ),
    index_dtype=helpers.get_dtypes("integer", full=False),
)
def test_jax_lax_argmin(
    *,
    dtype_x_axis,
    index_dtype,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        operand=x[0],
        axis=axis,
        index_dtype=index_dtype[0],
    )


# bitwise_xor
@handle_frontend_test(
    fn_tree="jax.lax.bitwise_xor",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_jax_lax_bitwise_xor(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        y=x[1],
    )


@handle_frontend_test(
    fn_tree="jax.lax.full_like",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=False, key="dtype")
    ),
    fill_val=_fill_value(),
    shape=st.one_of(helpers.get_shape() | st.none()),
)
def test_jax_lax_full_like(
    *,
    dtype_and_x,
    fill_val,
    shape,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtype, x = dtype_and_x
    fill_val = fill_val
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        fill_value=fill_val,
        dtype=dtype,
        shape=shape,
    )


@handle_frontend_test(
    fn_tree="jax.lax.exp",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_jax_lax_exp(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


@st.composite
def _sample_castable_numeric_dtype(draw):
    dtype = draw(helpers.get_dtypes("numeric", full=False, key="dtype"))[0]
    to_dtype = draw(helpers.get_dtypes("numeric", full=False))[0]
    assume(ivy.can_cast(dtype, to_dtype))
    return to_dtype


@handle_frontend_test(
    fn_tree="jax.lax.convert_element_type",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=False, key="dtype"),
        num_arrays=1,
        min_num_dims=1,
        min_value=-5,
        max_value=5,
    ),
    new_dtype=_sample_castable_numeric_dtype(),
)
def test_jax_lax_convert_element_type(
    *,
    dtype_and_x,
    new_dtype,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        operand=x[0],
        new_dtype=new_dtype,
    )


@handle_frontend_test(
    fn_tree="jax.lax.convert_element_type",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=5,
        valid_axis=True,
        allow_neg_axes=False,
        max_axes_size=1,
        force_int_axis=True,
    ),
    reverse=st.booleans(),
)
def test_jax_lax_cumprod(
    *,
    dtype_x_axis,
    reverse,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        operand=x[0],
        axis=axis,
        reverse=reverse,
    )


@handle_frontend_test(
    fn_tree="jax.lax.cumsum",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=5,
        valid_axis=True,
        allow_neg_axes=False,
        max_axes_size=1,
        force_int_axis=True,
    ),
    reverse=st.booleans(),
)
def test_jax_lax_cumsum(
    *,
    dtype_x_axis,
    reverse,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        operand=x[0],
        axis=axis,
        reverse=reverse,
    )


@handle_frontend_test(
    fn_tree="jax.lax.ge",
    dtypes_and_xs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_jax_lax_ge(
    *,
    dtypes_and_xs,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtypes, xs = dtypes_and_xs
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
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


@handle_frontend_test(
    fn_tree="jax.lax.reshape",
    x_reshape_permut=_reshape_helper(),
)
def test_jax_lax_reshape(
    *,
    x_reshape_permut,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    x, dtype, shape, dimensions = x_reshape_permut
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        operand=x[0],
        new_sizes=shape,
        dimensions=dimensions,
    )


@handle_frontend_test(
    fn_tree="jax.lax.broadcast",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
    sizes=helpers.get_shape(min_num_dims=1),
)
def test_jax_lax_broadcast(
    *,
    dtype_and_x,
    sizes,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        operand=x[0],
        sizes=sizes,
    )


@handle_frontend_test(
    fn_tree="jax.lax.reciprocal",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_jax_lax_reciprocal(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


@handle_frontend_test(
    fn_tree="jax.lax.sort",
    dtype_x_bounded_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
    ),
    is_stable=st.booleans(),
)
def test_jax_lax_sort(
    *,
    dtype_x_bounded_axis,
    is_stable,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x, axis = dtype_x_bounded_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        operand=x[0],
        dimension=axis,
        is_stable=is_stable,
    )


@handle_frontend_test(
    fn_tree="jax.lax.le",
    dtypes_and_xs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_jax_lax_le(
    *,
    dtypes_and_xs,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtypes, xs = dtypes_and_xs
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xs[0],
        y=xs[1],
    )


@handle_frontend_test(
    fn_tree="jax.lax.ne",
    dtypes_and_xs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_jax_lax_ne(
    *,
    dtypes_and_xs,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtypes, xs = dtypes_and_xs
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xs[0],
        y=xs[1],
    )


# cosh
@handle_frontend_test(
    fn_tree="jax.lax.cosh",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_jax_lax_cosh(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


@handle_frontend_test(
    fn_tree="jax.lax.add",
    dtypes_and_xs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_jax_lax_lt(
    *,
    dtypes_and_xs,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtypes, xs = dtypes_and_xs
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xs[0],
        y=xs[1],
    )


# round
@handle_frontend_test(
    fn_tree="jax.lax.round",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_jax_lax_round(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


@handle_frontend_test(
    fn_tree="jax.lax.pow",
    dtypes_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_jax_lax_pow(
    *,
    dtypes_and_values,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtypes, xs = dtypes_and_values
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xs[0],
        y=xs[1],
    )


@handle_frontend_test(
    fn_tree="jax.lax.gt",
    dtypes_and_xs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_jax_lax_gt(
    *,
    dtypes_and_xs,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtypes, xs = dtypes_and_xs
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xs[0],
        y=xs[1],
    )


@handle_frontend_test(
    fn_tree="jax.lax.int16",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_jax_lax_int16(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# cos
@handle_frontend_test(
    fn_tree="jax.lax.cos",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_jax_lax_cos(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


@st.composite
def _get_clamp_inputs(draw):
    shape = draw(
        helpers.get_shape(
            min_num_dims=1, max_num_dims=5, min_dim_size=2, max_dim_size=10
        )
    )
    x_dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
            shape=shape,
        )
    )

    min = draw(
        helpers.array_values(dtype=x_dtype[0], shape=shape, min_value=-25, max_value=0)
    )
    max = draw(
        helpers.array_values(dtype=x_dtype[0], shape=shape, min_value=1, max_value=25)
    )
    return x_dtype, x, min, max


@handle_frontend_test(
    fn_tree="jax.lax.clamp",
    dtype_x_min_max=_get_clamp_inputs(),
)
def test_jax_lax_clamp(
    *,
    dtype_x_min_max,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtypes, x, min_vals, max_vals = dtype_x_min_max
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        min=min_vals,
        x=x[0],
        max=max_vals,
    )


@handle_frontend_test(
    fn_tree="jax.lax.log",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=1,
    ),
)
def test_jax_lax_log(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


@handle_frontend_test(
    fn_tree="jax.lax.log1p",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        min_dim_size=2,
        valid_axis=True,
        force_int_axis=True,
        allow_neg_axes=False,
    ),
)
def test_jax_lax_rev(
    *,
    dtype_x_axis,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
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


@handle_frontend_test(
    fn_tree="jax.lax.div",
    dtypes_and_xs=_div_dtypes_and_xs(),
)
def test_jax_lax_div(
    *,
    dtypes_and_xs,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtypes, xs = dtypes_and_xs
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xs[0],
        y=xs[1],
    )


@handle_frontend_test(
    fn_tree="jax.lax.div",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_jax_lax_rsqrt(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-02,
        x=x[0],
    )


@handle_frontend_test(
    fn_tree="jax.lax.expm1",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_jax_lax_expm1(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# log1p
@handle_frontend_test(
    fn_tree="jax.lax.log1p",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_jax_lax_log1p(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
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


@handle_frontend_test(
    fn_tree="jax.lax.transpose",
    dtype_x_dims=_dtype_values_dims(),
)
def test_jax_lax_transpose(
    *,
    dtype_x_dims,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x, dims = dtype_x_dims
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
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


@handle_frontend_test(
    fn_tree="jax.lax.dot",
    dtypes_and_xs=_two_valid_xs(),
    dtype=helpers.get_dtypes("valid", full=False, none=True),
)
def test_jax_lax_dot(
    *,
    dtypes_and_xs,
    dtype,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtypes, xs = dtypes_and_xs
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        lhs=xs[0],
        rhs=xs[1],
        precision=None,
        preferred_element_type=dtype,
    )


@handle_frontend_test(
    fn_tree="jax.lax.conv",
    x_f_d_df=helpers.x_and_filters(dim=2),
)
def test_jax_lax_conv(
    *,
    x_f_d_df,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtype, x, filters, dilations, data_format, stride, pad = x_f_d_df
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        lhs=x[0],
        rhs=filters[0],
        window_strides=(stride, stride),
        padding=pad,
    )


@handle_frontend_test(
    fn_tree="jax.lax.conv_transpose",
    x_f_d_df=helpers.x_and_filters(dim=2),
)
def test_jax_lax_conv_transpose(
    *,
    x_f_d_df,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtype, x, filters, dilations, data_format, stride, pad = x_f_d_df
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        lhs=x[0],
        rhs=filters[0],
        strides=(stride, stride),
        padding=pad,
    )


@handle_frontend_test(
    fn_tree="jax.lax.sub",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_jax_lax_sub(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        y=x[1],
    )


@handle_frontend_test(
    fn_tree="jax.lax.rem",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
        allow_inf=False,
    ),
)
def test_jax_lax_rem(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
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
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        y=x[1],
    )


@handle_frontend_test(
    fn_tree="jax.lax.square",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric")
    ),
)
def test_jax_lax_square(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


@handle_frontend_test(
    fn_tree="jax.lax.erf",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", full=True)
    ),
)
def test_jax_lax_erf(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


@handle_frontend_test(
    fn_tree="jax.lax.shift_left",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_jax_lax_shift_left(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    # negative shifts will throw an exception
    # shifts >= dtype witdth produce backend-defined behavior
    x[1] = np.asarray(
        np.clip(x[1], 0, np.iinfo(input_dtype[1]).bits - 1), dtype=input_dtype[1]
    )

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        y=x[1],
    )


@handle_frontend_test(
    fn_tree="jax.lax.shift_right_logical",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_jax_lax_shift_right_logical(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    # negative shifts will throw an exception
    # shifts >= dtype witdth produce backend-defined behavior
    x[1] = np.asarray(
        np.clip(x[1], 0, np.iinfo(input_dtype[1]).bits - 1), dtype=input_dtype[1]
    )

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        y=x[1],
    )


# expand_dims
@handle_frontend_test(
    fn_tree="jax.lax.expand_dims",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=2,
        max_dim_size=10,
        force_int_axis=True,
        valid_axis=True,
    ),
)
def test_jax_lax_expand_dims(
    *,
    dtype_x_axis,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    x_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=x_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        array=x[0],
        dimensions=(axis,),
    )


# asinh
@handle_frontend_test(
    fn_tree="jax.lax.asinh",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_jax_lax_asinh(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# atanh
@handle_frontend_test(
    fn_tree="jax.lax.atanh",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_jax_lax_atanh(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )
