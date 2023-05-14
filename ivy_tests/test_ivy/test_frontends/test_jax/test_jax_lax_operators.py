# global
import numpy as np
import jax.numpy as jnp
from hypothesis import assume, strategies as st
import random
from jax.lax import ConvDimensionNumbers

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_tests.test_ivy.test_functional.test_nn.test_layers import (
    _assume_tf_dilation_gt_1,
)
from ivy.functional.frontends.jax.numpy import can_cast
from ivy.functional.frontends.jax.lax.operators import (
    _dimension_numbers,
    _argsort_tuple,
)


# add
@handle_frontend_test(
    fn_tree="jax.lax.add",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_jax_lax_add(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        y=x[1],
    )


# tan
@handle_frontend_test(
    fn_tree="jax.lax.tan",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    test_with_out=st.just(False),
)
def test_jax_lax_tan(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# max
@handle_frontend_test(
    fn_tree="jax.lax.max",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_jax_lax_max(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
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
        helpers.list_of_size(
            x=helpers.ints(min_value=2, max_value=4),
            size=num_dims - 1,
        )
    )
    unique_idx = draw(helpers.ints(min_value=0, max_value=num_dims - 1))
    unique_dims = draw(
        helpers.list_of_size(
            x=helpers.ints(min_value=2, max_value=3),
            size=num_arrays,
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
    test_with_out=st.just(False),
)
def test_jax_lax_concat(
    *,
    xs_n_input_dtypes_n_unique_idx,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    xs, input_dtypes, unique_idx = xs_n_input_dtypes_n_unique_idx
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
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
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        frontend=frontend,
        test_flags=test_flags,
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
    test_with_out=st.just(False),
)
def test_jax_lax_abs(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# sqrt
@handle_frontend_test(
    fn_tree="jax.lax.sqrt",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    test_with_out=st.just(False),
)
def test_jax_lax_sqrt(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# acos
@handle_frontend_test(
    fn_tree="jax.lax.acos",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    test_with_out=st.just(False),
)
def test_jax_lax_acos(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# sin
@handle_frontend_test(
    fn_tree="jax.lax.sin",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    test_with_out=st.just(False),
)
def test_jax_lax_sin(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
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
    test_with_out=st.just(False),
)
def test_jax_lax_sign(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# asin
@handle_frontend_test(
    fn_tree="jax.lax.asin",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    test_with_out=st.just(False),
)
def test_jax_lax_asin(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# sinh
@handle_frontend_test(
    fn_tree="jax.lax.sinh",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    test_with_out=st.just(False),
)
def test_jax_lax_sinh(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
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
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
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
    test_with_out=st.just(False),
)
def test_jax_lax_min(
    *,
    dtypes_and_xs,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtypes, xs = dtypes_and_xs
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
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
    test_with_out=st.just(False),
)
def test_jax_lax_eq(
    *,
    dtypes_and_xs,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtypes, xs = dtypes_and_xs
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
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
    test_with_out=st.just(False),
)
def test_jax_lax_mul(
    *,
    dtypes_and_xs,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtypes, xs = dtypes_and_xs
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xs[0],
        y=xs[1],
    )


# atan
@handle_frontend_test(
    fn_tree="jax.lax.atan",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    test_with_out=st.just(False),
)
def test_jax_lax_atan(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


@handle_frontend_test(
    fn_tree="jax.lax.ceil",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_with_out=st.just(False),
)
def test_jax_lax_ceil(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
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
    test_with_out=st.just(False),
)
def test_jax_lax_bitwise_and(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
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
    test_with_out=st.just(False),
)
def test_jax_lax_bitwise_or(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
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
    test_with_out=st.just(False),
)
def test_jax_lax_bitwise_not(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


@handle_frontend_test(
    fn_tree="jax.lax.neg",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("signed_integer"),
    ),
    test_with_out=st.just(False),
)
def test_jax_lax_neg(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
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
    test_with_out=st.just(False),
)
def test_jax_lax_argmax(
    *,
    dtype_x_axis,
    index_dtype,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
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
    test_with_out=st.just(False),
)
def test_jax_lax_argmin(
    *,
    dtype_x_axis,
    index_dtype,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
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
    test_with_out=st.just(False),
)
def test_jax_lax_bitwise_xor(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
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
    dtype=st.shared(helpers.get_dtypes("numeric", full=False), key="dtype"),
    test_with_out=st.just(False),
)
def test_jax_lax_full_like(
    *,
    dtype_and_x,
    fill_val,
    shape,
    dtype,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    fill_val = fill_val
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
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
    test_with_out=st.just(False),
)
def test_jax_lax_exp(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


@handle_frontend_test(
    fn_tree="jax.lax.convert_element_type",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    new_dtype=helpers.get_dtypes("valid", full=False),
    test_with_out=st.just(False),
)
def test_jax_lax_convert_element_type(
    *,
    dtype_and_x,
    new_dtype,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    assume(can_cast(input_dtype[0], new_dtype[0]))
    helpers.test_frontend_function(
        input_dtypes=input_dtype + new_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        operand=x[0],
        new_dtype=new_dtype[0],
    )


@handle_frontend_test(
    fn_tree="jax.lax.cumprod",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=5,
        min_value=-5,
        max_value=5,
        valid_axis=True,
        allow_neg_axes=False,
        max_axes_size=1,
        force_int_axis=True,
    ),
    reverse=st.booleans(),
    test_with_out=st.just(False),
)
def test_jax_lax_cumprod(
    *,
    dtype_x_axis,
    reverse,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-2,
        atol=1e-2,
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
    test_with_out=st.just(False),
)
def test_jax_lax_cumsum(
    *,
    dtype_x_axis,
    reverse,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
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
    test_with_out=st.just(False),
)
def test_jax_lax_ge(
    *,
    dtypes_and_xs,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtypes, xs = dtypes_and_xs
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xs[0],
        y=xs[1],
    )


@st.composite
def _reshape_helper(draw):
    # generate a shape s.t len(shape) > 0
    shape = draw(
        helpers.get_shape(
            allow_none=False,
            min_num_dims=1,
            max_num_dims=3,
            min_dim_size=1,
            max_dim_size=3,
        )
    )

    reshape_shape = draw(helpers.reshape_shapes(shape=shape))

    dtypes, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"),
            shape=shape,
        )
    )
    is_dim = draw(st.booleans())
    if is_dim:
        dims = [x for x in range(len(shape))]
        permut = draw(st.permutations(dims))
        return x, dtypes, reshape_shape, permut
    else:
        return x, dtypes, reshape_shape, None


@handle_frontend_test(
    fn_tree="jax.lax.reshape",
    x_reshape_permut=_reshape_helper(),
    test_with_out=st.just(False),
)
def test_jax_lax_reshape(
    *,
    x_reshape_permut,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    x, dtype, shape, dimensions = x_reshape_permut
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
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
    test_with_out=st.just(False),
)
def test_jax_lax_broadcast(
    *,
    dtype_and_x,
    sizes,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
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
    test_with_out=st.just(False),
)
def test_jax_lax_reciprocal(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
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
    test_with_out=st.just(False),
)
def test_jax_lax_sort(
    *,
    dtype_x_bounded_axis,
    is_stable,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x, axis = dtype_x_bounded_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
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
    test_with_out=st.just(False),
)
def test_jax_lax_le(
    *,
    dtypes_and_xs,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtypes, xs = dtypes_and_xs
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
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
    test_with_out=st.just(False),
)
def test_jax_lax_ne(
    *,
    dtypes_and_xs,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtypes, xs = dtypes_and_xs
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xs[0],
        y=xs[1],
    )


# cosh
@handle_frontend_test(
    fn_tree="jax.lax.cosh",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    test_with_out=st.just(False),
)
def test_jax_lax_cosh(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
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
    test_with_out=st.just(False),
)
def test_jax_lax_lt(
    *,
    dtypes_and_xs,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtypes, xs = dtypes_and_xs
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xs[0],
        y=xs[1],
    )


# round
@handle_frontend_test(
    fn_tree="jax.lax.round",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    rounding_method=st.sampled_from([0, 1]),
    test_with_out=st.just(False),
)
def test_jax_lax_round(
    *,
    dtype_and_x,
    rounding_method,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        rounding_method=rounding_method,
    )


@handle_frontend_test(
    fn_tree="jax.lax.pow",
    dtypes_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_jax_lax_pow(
    *,
    dtypes_and_values,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtypes, xs = dtypes_and_values
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xs[0],
        y=xs[1],
    )


@st.composite
def _pad_helper(draw):
    dtype, x, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("bool"),
            ret_shape=True,
            min_num_dims=1,
            min_dim_size=2,
            min_value=-100,
            max_value=100,
        ).filter(lambda _x: _x[0][0] not in ["float16", "bfloat16"])
    )
    ndim = len(shape)
    min_dim = min(shape)
    padding_config = draw(
        st.lists(
            st.tuples(
                st.integers(min_value=-(min_dim - 1), max_value=min_dim - 1),
                st.integers(min_value=-(min_dim - 1), max_value=min_dim - 1),
                st.integers(min_value=0, max_value=min_dim - 1),
            ),
            min_size=ndim,
            max_size=ndim,
        )
    )
    padding_value = draw(st.booleans())
    return dtype, x[0], padding_value, padding_config


@handle_frontend_test(
    fn_tree="jax.lax.pad",
    dtype_x_params=_pad_helper(),
    test_with_out=st.just(False),
)
def test_jax_lax_pad(
    *,
    dtype_x_params,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    dtype, operand, padding_value, padding_config = dtype_x_params
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        operand=operand,
        padding_value=padding_value,
        padding_config=padding_config,
    )


@handle_frontend_test(
    fn_tree="jax.lax.gt",
    dtypes_and_xs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_jax_lax_gt(
    *,
    dtypes_and_xs,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtypes, xs = dtypes_and_xs
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xs[0],
        y=xs[1],
    )


# cos
@handle_frontend_test(
    fn_tree="jax.lax.cos",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    test_with_out=st.just(False),
)
def test_jax_lax_cos(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
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
    test_with_out=st.just(False),
)
def test_jax_lax_clamp(
    *,
    dtype_x_min_max,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtypes, x, min_vals, max_vals = dtype_x_min_max
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
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
    test_with_out=st.just(False),
)
def test_jax_lax_log(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


@handle_frontend_test(
    fn_tree="jax.lax.rev",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        min_dim_size=2,
        valid_axis=True,
        force_int_axis=True,
        allow_neg_axes=False,
    ),
    test_with_out=st.just(False),
)
def test_jax_lax_rev(
    *,
    dtype_x_axis,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
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
        helpers.array_values(dtype=dtype[0], min_value=-20, max_value=20, shape=shape)
    )
    return dtype, [dividend[0], divisor]


@handle_frontend_test(
    fn_tree="jax.lax.div",
    dtypes_and_xs=_div_dtypes_and_xs(),
    test_with_out=st.just(False),
)
def test_jax_lax_div(
    *,
    dtypes_and_xs,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtypes, xs = dtypes_and_xs
    assume(not np.any(np.isclose(xs[1], 0)))
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xs[0],
        y=xs[1],
    )


@handle_frontend_test(
    fn_tree="jax.lax.rsqrt",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_with_out=st.just(False),
)
def test_jax_lax_rsqrt(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
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
    test_with_out=st.just(False),
)
def test_jax_lax_expm1(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# log1p
@handle_frontend_test(
    fn_tree="jax.lax.log1p",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        small_abs_safety_factor=2,
        safety_factor_scale="log",
    ),
    test_with_out=st.just(False),
)
def test_jax_lax_log1p(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
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
    test_with_out=st.just(False),
)
def test_jax_lax_transpose(
    *,
    dtype_x_dims,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x, dims = dtype_x_dims
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        operand=x[0],
        permutation=dims,
    )


@st.composite
def _get_dtype_inputs_for_dot(draw):
    dim_size = draw(helpers.ints(min_value=1, max_value=5))
    dtype = draw(helpers.get_dtypes("numeric", index=1, full=False))
    if dim_size == 1:
        lhs = draw(
            helpers.array_values(
                dtype=dtype[0], shape=(dim_size,), min_value=2, max_value=5
            )
        )
        rhs = draw(
            helpers.array_values(
                dtype=dtype[0], shape=(dim_size,), min_value=2, max_value=5
            )
        )
    else:
        lhs = draw(
            helpers.array_values(
                dtype=dtype[0], shape=(dim_size, dim_size), min_value=2, max_value=5
            )
        )
        rhs = draw(
            helpers.array_values(
                dtype=dtype[0], shape=(dim_size, dim_size), min_value=2, max_value=5
            )
        )
    is_pref = draw(st.booleans())
    if is_pref:
        dtype, values, pref = draw(
            helpers.get_castable_dtype(
                draw(helpers.get_dtypes("numeric")), dtype[0], [lhs, rhs]
            )
        )
        assume(can_cast(dtype, pref))
        return [dtype], pref, values[0], values[1]
    else:
        return dtype, None, lhs, rhs


@handle_frontend_test(
    fn_tree="jax.lax.dot",
    dtypes_and_xs=_get_dtype_inputs_for_dot(),
    test_with_out=st.just(False),
)
def test_jax_lax_dot(
    *,
    dtypes_and_xs,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtypes, dtype, lhs, rhs = dtypes_and_xs
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-2,
        atol=1e-2,
        lhs=lhs,
        rhs=rhs,
        precision=None,
        preferred_element_type=dtype,
    )


@st.composite
def _general_dot_helper(draw):
    input_dtype = draw(helpers.get_dtypes("numeric", full=False))
    lshape = draw(
        st.lists(st.integers(min_value=1, max_value=10), min_size=2, max_size=52)
    )
    ndims = len(lshape)
    perm_id = random.sample(list(range(ndims)), ndims)
    rshape = [lshape[i] for i in perm_id]
    ldtype, lhs = draw(
        helpers.dtype_and_values(
            dtype=input_dtype,
            min_value=-1e04,
            max_value=1e04,
            shape=lshape,
        )
    )
    rdtype, rhs = draw(
        helpers.dtype_and_values(
            dtype=input_dtype,
            min_value=-1e04,
            max_value=1e04,
            shape=rshape,
        )
    )
    ind_list = list(range(ndims))
    batch_n = draw(st.integers(min_value=1, max_value=len(lshape) - 1))
    lhs_batch = random.sample(ind_list, batch_n)
    rhs_batch = [perm_id.index(i) for i in lhs_batch]
    lhs_contracting = [i for i in ind_list if i not in lhs_batch]
    rhs_contracting = [perm_id.index(i) for i in lhs_contracting]
    is_pref = draw(st.booleans())
    if is_pref:
        dtype, pref = draw(
            helpers.get_castable_dtype(
                draw(helpers.get_dtypes("numeric")), input_dtype[0]
            )
        )
        assume(can_cast(dtype, pref))
        pref_dtype = pref
    else:
        pref_dtype = None
    return (
        ldtype + rdtype,
        (lhs[0], rhs[0]),
        ((lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch)),
        pref_dtype,
    )


@handle_frontend_test(
    fn_tree="jax.lax.dot_general",
    dtypes_lr_dims=_general_dot_helper(),
    test_with_out=st.just(False),
)
def test_jax_lax_dot_general(
    *,
    dtypes_lr_dims,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    dtypes, lr, dims, dtype = dtypes_lr_dims
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        lhs=lr[0],
        rhs=lr[1],
        dimension_numbers=dims,
        precision=None,
        preferred_element_type=dtype,
    )


@st.composite
def x_and_filters(draw, dim=2, transpose=False, general=False):
    if not isinstance(dim, int):
        dim = draw(dim)
    batch_size = draw(st.integers(1, 5))
    filter_shape = draw(
        helpers.get_shape(
            min_num_dims=dim, max_num_dims=dim, min_dim_size=1, max_dim_size=5
        )
    )
    dtype = draw(helpers.get_dtypes("float", full=False))
    padding = draw(
        st.one_of(
            st.lists(
                st.tuples(
                    st.integers(min_value=0, max_value=3),
                    st.integers(min_value=0, max_value=3),
                ),
                min_size=dim,
                max_size=dim,
            ),
            st.sampled_from(["SAME", "VALID"]),
        )
    )
    input_channels = draw(st.integers(1, 3))
    output_channels = draw(st.integers(1, 3))
    group_list = [i for i in range(1, 6)]
    if not transpose:
        group_list = list(filter(lambda x: (input_channels % x == 0), group_list))
    else:
        group_list = list(filter(lambda x: (output_channels % x == 0), group_list))
    fc = draw(st.sampled_from(group_list)) if general else 1
    strides = draw(st.lists(st.integers(1, 3), min_size=dim, max_size=dim))
    dilations = draw(st.lists(st.integers(1, 3), min_size=dim, max_size=dim))
    if general:
        if dim == 2:
            dim_num_st1 = st.sampled_from(["NCHW", "NHWC"])
            dim_num_st2 = st.sampled_from(["OIHW", "HWIO"])
        elif dim == 1:
            dim_num_st1 = st.sampled_from(["NWC", "NCW"])
            dim_num_st2 = st.sampled_from(["OIW", "WIO"])
        else:
            dim_num_st1 = st.sampled_from(["NDHWC", "NCDHW"])
            dim_num_st2 = st.sampled_from(["OIDHW", "DHWIO"])
        dim_seq = [*range(0, dim + 2)]
        dimension_numbers = draw(
            st.sampled_from(
                [
                    None,
                    (draw(dim_num_st1), draw(dim_num_st2), draw(dim_num_st1)),
                    ConvDimensionNumbers(
                        *map(
                            tuple,
                            draw(
                                st.lists(
                                    st.permutations(dim_seq), min_size=3, max_size=3
                                )
                            ),
                        )
                    ),
                ]
            )
        )
    else:
        dimension_numbers = (
            ("NCH", "OIH", "NCH")
            if dim == 1
            else ("NCHW", "OIHW", "NCHW") if dim == 2 else ("NCDHW", "OIDHW", "NCDHW")
        )
    dim_nums = _dimension_numbers(dimension_numbers, dim + 2, transp=transpose)
    if not transpose:
        output_channels = output_channels * fc
        channel_shape = (output_channels, input_channels // fc)
    else:
        input_channels = input_channels * fc
        channel_shape = (output_channels // fc, input_channels)
    x_dim = []
    for i in range(dim):
        min_x = filter_shape[i] + (filter_shape[i] - 1) * (dilations[i] - 1)
        x_dim.append(draw(st.integers(min_x, min_x + 1)))
    x_shape = (batch_size, input_channels, *x_dim)
    filter_shape = channel_shape + filter_shape
    vals = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=x_shape,
            min_value=0.0,
            max_value=1.0,
        )
    )
    vals = ivy.permute_dims(vals, axes=_argsort_tuple(dim_nums[0]))
    filters = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=filter_shape,
            min_value=0.0,
            max_value=1.0,
        )
    )
    filters = ivy.permute_dims(filters, axes=_argsort_tuple(dim_nums[1]))
    if general and not transpose:
        x_dilation = draw(st.lists(st.integers(1, 3), min_size=dim, max_size=dim))
        dilations = (dilations, x_dilation)
    if draw(st.booleans()):
        p_dtype, pref = draw(
            helpers.get_castable_dtype(draw(helpers.get_dtypes("float")), dtype[0])
        )
        assume(can_cast(p_dtype, pref))
    else:
        pref = None
    return (
        dtype,
        vals,
        filters,
        dilations,
        dimension_numbers,
        strides,
        padding,
        fc,
        pref,
    )


@handle_frontend_test(
    fn_tree="jax.lax.conv",
    x_f_d_other=x_and_filters(),
    test_with_out=st.just(False),
)
def test_jax_lax_conv(
    *,
    x_f_d_other,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    dtype, x, filters, dilation, dim_num, stride, pad, fc, pref = x_f_d_other
    _assume_tf_dilation_gt_1(ivy.current_backend_str(), on_device, dilation)
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        lhs=x,
        rhs=filters,
        window_strides=stride,
        padding=pad,
        precision=None,
        preferred_element_type=pref,
    )


@handle_frontend_test(
    fn_tree="jax.lax.conv_transpose",
    x_f_d_other=x_and_filters(general=True, transpose=True),
    test_with_out=st.just(False),
)
def test_jax_lax_conv_transpose(
    *,
    x_f_d_other,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    dtype, x, filters, dilation, dim_num, stride, pad, fc, pref = x_f_d_other
    _assume_tf_dilation_gt_1(ivy.current_backend_str(), on_device, dilation)
    helpers.test_frontend_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        lhs=x,
        rhs=filters,
        strides=stride,
        padding=pad,
        rhs_dilation=dilation,
        dimension_numbers=dim_num,
        transpose_kernel=False,
        precision=None,
        preferred_element_type=pref,
    )


@handle_frontend_test(
    fn_tree="jax.lax.conv_general_dilated",
    x_f_d_other=x_and_filters(general=True),
    test_with_out=st.just(False),
)
def test_jax_lax_conv_general_dilated(
    *,
    x_f_d_other,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    dtype, x, filters, dilations, dim_num, stride, pad, fc, pref = x_f_d_other
    _assume_tf_dilation_gt_1(ivy.current_backend_str(), on_device, dilations[0])
    assume(
        not (isinstance(pad, str) and not len(dilations[1]) == dilations[1].count(1))
    )
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        lhs=x,
        rhs=filters,
        window_strides=stride,
        padding=pad,
        lhs_dilation=dilations[1],
        rhs_dilation=dilations[0],
        dimension_numbers=dim_num,
        feature_group_count=fc,
        batch_group_count=1,
        precision=None,
        preferred_element_type=pref,
    )


@handle_frontend_test(
    fn_tree="jax.lax.sub",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_jax_lax_sub(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
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
    test_with_out=st.just(False),
)
def test_jax_lax_rem(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x[0], 0)))  # ToDO, should use safety factor?
    assume(not np.any(np.isclose(x[1], 0)))
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
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
    test_with_out=st.just(False),
)
def test_jax_lax_square(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


@handle_frontend_test(
    fn_tree="jax.lax.erf",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    test_with_out=st.just(False),
)
def test_jax_lax_erf(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-2,
        atol=1e-2,
        x=x[0],
    )


@handle_frontend_test(
    fn_tree="jax.lax.shift_left",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_jax_lax_shift_left(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    # negative shifts will throw an exception
    # shifts >= dtype witdth produce backend-defined behavior
    x[1] = np.asarray(
        np.clip(x[1], 0, np.iinfo(input_dtype[1]).bits - 1), dtype=input_dtype[1]
    )

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
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
        min_value=0,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_jax_lax_shift_right_logical(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    # negative shifts will throw an exception
    # shifts >= dtype witdth produce backend-defined behavior
    x[1] = np.asarray(
        np.clip(x[1], 0, np.iinfo(input_dtype[1]).bits - 1), dtype=input_dtype[1]
    )

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        y=x[1],
    )


@st.composite
def _slice_helper(draw):
    dtype, x, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            min_num_dims=1,
            ret_shape=True,
        ),
    )
    start_indices, limit_indices, strides = [], [], []
    for i in shape:
        start_indices += [draw(st.integers(min_value=0, max_value=i - 1))]
        limit_indices += [
            draw(
                st.integers(min_value=0, max_value=i - 1).filter(
                    lambda _x: _x > start_indices[-1]
                )
            )
        ]
        strides += [draw(st.integers(min_value=1, max_value=i))]
    return dtype, x, start_indices, limit_indices, strides


@handle_frontend_test(
    fn_tree="jax.lax.slice",
    dtype_x_params=_slice_helper(),
    test_with_out=st.just(False),
)
def test_jax_lax_slice(
    *,
    dtype_x_params,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    dtype, x, start_indices, limit_indices, strides = dtype_x_params
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        operand=x[0],
        start_indices=start_indices,
        limit_indices=limit_indices,
        strides=strides,
    )


@st.composite
def _slice_in_dim_helper(draw):
    dtype, x, axis = draw(
        helpers.dtype_values_axis(
            available_dtypes=helpers.get_dtypes("valid"),
            min_num_dims=1,
            force_int_axis=True,
            valid_axis=True,
        ),
    )
    operand = x[0]
    start_index = draw(
        st.integers(min_value=-abs(operand.shape[axis]), max_value=operand.shape[axis])
    )
    if start_index < 0:
        limit_index = draw(
            st.integers(
                min_value=start_index + operand.shape[axis],
                max_value=operand.shape[axis],
            )
        )
    else:
        limit_index = draw(
            st.integers(
                min_value=-abs(operand.shape[axis]), max_value=operand.shape[axis]
            ).filter(lambda _x: _x >= start_index)
        )
    stride = draw(st.integers(min_value=1, max_value=abs(limit_index + 1)))
    return dtype, x, start_index, limit_index, stride, axis


@handle_frontend_test(
    fn_tree="jax.lax.slice_in_dim",
    dtype_x_params=_slice_in_dim_helper(),
    test_with_out=st.just(False),
)
def test_jax_lax_slice_in_dim(
    *,
    dtype_x_params,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    dtype, x, start_index, limit_index, stride, axis = dtype_x_params
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        operand=x[0],
        start_index=start_index,
        limit_index=limit_index,
        stride=stride,
        axis=axis,
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
    test_with_out=st.just(False),
)
def test_jax_lax_expand_dims(
    *,
    dtype_x_axis,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    x_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=x_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        array=x[0],
        dimensions=(axis,),
    )


# asinh
@handle_frontend_test(
    fn_tree="jax.lax.asinh",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    test_with_out=st.just(False),
)
def test_jax_lax_asinh(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# atanh
@handle_frontend_test(
    fn_tree="jax.lax.atanh",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    test_with_out=st.just(False),
)
def test_jax_lax_atanh(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# select
@st.composite
def _dtype_pred_ontrue_on_false(draw):
    shape = draw(helpers.get_shape(min_num_dims=1, min_dim_size=1))
    pred = draw(helpers.array_values(dtype="bool", shape=shape))
    dtypes, on_true_on_false = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"),
            num_arrays=2,
            shape=shape,
            large_abs_safety_factor=16,
            small_abs_safety_factor=16,
            safety_factor_scale="log",
            shared_dtype=True,
        )
    )
    return dtypes, pred, on_true_on_false


@handle_frontend_test(
    fn_tree="jax.lax.select",
    dtype_pred_ontrue_on_false=_dtype_pred_ontrue_on_false(),
    test_with_out=st.just(False),
)
def test_jax_lax_select(
    *,
    dtype_pred_ontrue_on_false,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, pred, on_true_on_false = dtype_pred_ontrue_on_false
    helpers.test_frontend_function(
        input_dtypes=["bool"] + input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        pred=pred,
        on_true=on_true_on_false[0],
        on_false=on_true_on_false[0],
    )


# top_k
@handle_frontend_test(
    fn_tree="jax.lax.top_k",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
        min_dim_size=4,
        max_dim_size=10,
    ),
    k=helpers.ints(min_value=1, max_value=4),
    test_with_out=st.just(False),
)
def test_jax_lax_top_k(
    *,
    dtype_and_x,
    k,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        operand=x[0],
        k=k,
        # test_values=False,
    )


@st.composite
def _reduce_window_helper(draw):
    dtype = draw(
        helpers.get_dtypes("numeric", full=False).filter(
            lambda x: x[0] not in ("bfloat16", "uint64", "uint32")
        )
    )
    init_value = draw(helpers.array_values(dtype=dtype[0], shape=()))

    def py_func(accumulator, window):
        if not len(window.shape):
            window = jnp.expand_dims(window, 0)
        sum = 0
        for w in window:
            sum += w
        return accumulator + sum

    ndim = draw(st.integers(min_value=1, max_value=4))

    _, others = draw(
        helpers.dtype_and_values(
            num_arrays=4,
            dtype=["int64"] * 4,
            shape=(ndim,),
            min_value=1,
            max_value=3,
            small_abs_safety_factor=1,
            large_abs_safety_factor=1,
        )
    )
    others = [other.tolist() for other in others]

    window, dilation = others[0], others[2]
    op_shape = []
    for i in range(ndim):
        min_x = window[i] + (window[i] - 1) * (dilation[i] - 1)
        op_shape.append(draw(st.integers(min_x, min_x + 1)))
    dtype, operand = draw(
        helpers.dtype_and_values(
            dtype=dtype,
            shape=op_shape,
        )
    )

    padding = draw(
        st.one_of(
            st.lists(
                st.tuples(
                    st.integers(min_value=0, max_value=3),
                    st.integers(min_value=0, max_value=3),
                ),
                min_size=ndim,
                max_size=ndim,
            ),
            st.sampled_from(["SAME", "VALID"]),
        )
    )

    return dtype * 2, operand, init_value, py_func, others, padding


@handle_frontend_test(
    fn_tree="jax.lax.reduce_window",
    all_args=_reduce_window_helper(),
    test_with_out=st.just(False),
)
def test_jax_lax_reduce_window(
    *,
    all_args,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    dtypes, operand, init_value, computation, others, padding = all_args
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        operand=operand[0],
        init_value=init_value,
        computation=computation,
        window_dimensions=others[0],
        window_strides=others[1],
        padding=padding,
        base_dilation=others[2],
        window_dilation=None,
    )


# real
@handle_frontend_test(
    fn_tree="jax.lax.real",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("complex")
    ),
)
def test_jax_lax_real(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=True,
        x=x[0],
    )


# squeeze
@st.composite
def _squeeze_helper(draw):
    shape = draw(st.shared(helpers.get_shape(), key="value_shape"))
    valid_axes = []
    for index, axis in enumerate(shape):
        if axis == 1:
            valid_axes.append(index)
    return valid_axes


@handle_frontend_test(
    fn_tree="jax.lax.squeeze",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=st.shared(
            helpers.get_shape(
                allow_none=False,
                min_num_dims=1,
                max_num_dims=10,
                min_dim_size=1,
                max_dim_size=5,
            ),
            key="value_shape",
        ),
    ),
    dim=_squeeze_helper(),
)
def test_jax_lax_squeeze(
    *,
    dtype_and_values,
    dim,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, value = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        array=value[0],
        dimensions=dim,
    )


# nextafter
@handle_frontend_test(
    fn_tree="jax.lax.nextafter",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=["float32", "float64"],
        min_value=-100,
        max_value=100,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_jax_lax_nextafter(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=x[0],
        x2=x[0],
    )


# conj
@handle_frontend_test(
    fn_tree="jax.lax.conj",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=["complex64"],
    ),
)
def test_jax_lax_conj(
    *,
    dtype_and_x,
    test_flags,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )
