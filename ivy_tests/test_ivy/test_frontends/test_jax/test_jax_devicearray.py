# global
from hypothesis import given, strategies as st

# local
import jax.numpy as jnp
from ivy.functional.frontends.jax.devicearray import DeviceArray
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# reshape
@st.composite
def _reshape_helper(draw):
    # generate a shape s.t len(shape) > 0
    shape = draw(helpers.get_shape(min_num_dims=1))

    reshape_shape = draw(helpers.reshape_shapes(shape=shape))

    dtype = draw(helpers.array_dtypes(num_arrays=1))[0]
    x = draw(helpers.array_values(dtype=dtype, shape=shape))

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
        fn_name="ivy.functional.frontends.jax.DeviceArray.reshape"
    ),
)
def test_jax_instance_reshape(
    x_reshape_permut,
    as_variable,
    num_positional_args,
    native_array,
):
    x, dtype, shape, dimensions = x_reshape_permut
    helpers.test_frontend_array_instance_method(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="jax",
        frontend_class=DeviceArray,
        fn_tree="DeviceArray.reshape",
        self=x[0],
        new_sizes=shape,
        dimensions=dimensions,
    )


# add
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.DeviceArray.add",
    ),
)
def test_jax_instance_add(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_array_instance_method(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="jax",
        frontend_class=DeviceArray,
        fn_tree="DeviceArray.add",
        self=x[0],
        other=x[1],
    )


# __pos__
@handle_cmd_line_args
@given(dtype_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("numeric")))
def test_jax_special_pos(
    dtype_x,
    fw,
):
    input_dtype, x = dtype_x
    ret = +DeviceArray(x[0])
    ret_gt = +jnp.array(x[0], dtype=input_dtype[0])
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    for (u, v) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="jax",
        )


# __neg__
@handle_cmd_line_args
@given(dtype_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("numeric")))
def test_jax_special_neg(
    dtype_x,
    fw,
):
    input_dtype, x = dtype_x
    ret = -DeviceArray(x[0])
    ret_gt = -jnp.array(x[0], dtype=input_dtype[0])
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    for (u, v) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="jax",
        )


# __eq__
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"), num_arrays=2
    )
)
def test_jax_special_eq(
    dtype_x,
    fw,
):
    input_dtype, x = dtype_x
    ret = DeviceArray(x[0]) == DeviceArray(x[1])
    ret_gt = jnp.array(x[0], dtype=input_dtype[0]) == jnp.array(
        x[1], dtype=input_dtype[1]
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    for (u, v) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="jax",
        )


# __ne__
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"), num_arrays=2
    )
)
def test_jax_special_ne(
    dtype_x,
    fw,
):
    input_dtype, x = dtype_x
    ret = DeviceArray(x[0]) != DeviceArray(x[1])
    ret_gt = jnp.array(x[0], dtype=input_dtype[0]) != jnp.array(
        x[1], dtype=input_dtype[1]
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    for (u, v) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="jax",
        )


# __lt__
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"), num_arrays=2
    )
)
def test_jax_special_lt(
    dtype_x,
    fw,
):
    input_dtype, x = dtype_x
    ret = DeviceArray(x[0]) < DeviceArray(x[1])
    ret_gt = jnp.array(x[0], dtype=input_dtype[0]) < jnp.array(
        x[1], dtype=input_dtype[1]
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    for (u, v) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="jax",
        )


# __le__
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"), num_arrays=2
    )
)
def test_jax_special_le(
    dtype_x,
    fw,
):
    input_dtype, x = dtype_x
    ret = DeviceArray(x[0]) <= DeviceArray(x[1])
    ret_gt = jnp.array(x[0], dtype=input_dtype[0]) <= jnp.array(
        x[1], dtype=input_dtype[1]
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    for (u, v) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="jax",
        )


# __gt__
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"), num_arrays=2
    )
)
def test_jax_special_gt(
    dtype_x,
    fw,
):
    input_dtype, x = dtype_x
    ret = DeviceArray(x[0]) > DeviceArray(x[1])
    ret_gt = jnp.array(x[0], dtype=input_dtype[0]) > jnp.array(
        x[1], dtype=input_dtype[1]
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    for (u, v) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="jax",
        )


# __ge__
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"), num_arrays=2
    )
)
def test_jax_special_ge(
    dtype_x,
    fw,
):
    input_dtype, x = dtype_x
    ret = DeviceArray(x[0]) >= DeviceArray(x[1])
    ret_gt = jnp.array(x[0], dtype=input_dtype[0]) >= jnp.array(
        x[1], dtype=input_dtype[1]
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    for (u, v) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="jax",
        )


# __abs__
@handle_cmd_line_args
@given(dtype_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("numeric")))
def test_jax_special_abs(
    dtype_x,
    fw,
):
    input_dtype, x = dtype_x
    ret = abs(DeviceArray(x[0]))
    ret_gt = abs(jnp.array(x[0], dtype=input_dtype[0]))
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    for (u, v) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="jax",
        )


@st.composite
def _get_dtype_x_and_int(draw, *, dtype="numeric"):
    x_dtype, x = draw(
        helpers.dtype_and_values(available_dtypes=helpers.get_dtypes(dtype))
    )
    x_int = draw(helpers.ints(min_value=0, max_value=10))
    return x_dtype, x, x_int


# __pow__
@handle_cmd_line_args
@given(dtype_x_pow=_get_dtype_x_and_int())
def test_jax_special_pow(
    dtype_x_pow,
    fw,
):
    x_dtype, x, pow = dtype_x_pow
    ret = DeviceArray(x[0]) ** pow
    ret_gt = jnp.array(x[0], dtype=x_dtype[0]) ** pow
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    for (u, v) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="jax",
        )


# __rpow__
@handle_cmd_line_args
@given(dtype_x_pow=_get_dtype_x_and_int())
def test_jax_special_rpow(
    dtype_x_pow,
    fw,
):
    x_dtype, x, pow = dtype_x_pow
    ret = DeviceArray(pow).__rpow__(DeviceArray(x[0]))
    ret_gt = jnp.array(pow).__rpow__(jnp.array(x[0], dtype=x_dtype[0]))
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    for (u, v) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="jax",
        )


# __and__
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("signed_integer"), num_arrays=2
    )
)
def test_jax_special_and(
    dtype_x,
    fw,
):
    input_dtype, x = dtype_x
    ret = DeviceArray(x[0]) & DeviceArray(x[1])
    ret_gt = jnp.array(x[0], dtype=input_dtype[0]) & jnp.array(
        x[1], dtype=input_dtype[1]
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    for (u, v) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="jax",
        )


# __rand__
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("signed_integer"), num_arrays=2
    )
)
def test_jax_special_rand(
    dtype_x,
    fw,
):
    input_dtype, x = dtype_x
    ret = DeviceArray(x[1]).__rand__(DeviceArray(x[0]))
    ret_gt = jnp.array(x[1], dtype=input_dtype[1]).__rand__(
        jnp.array(x[0], dtype=input_dtype[0])
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    for (u, v) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="jax",
        )


# __or__
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("signed_integer"), num_arrays=2
    )
)
def test_jax_special_or(
    dtype_x,
    fw,
):
    input_dtype, x = dtype_x
    ret = DeviceArray(x[0]) | DeviceArray(x[1])
    ret_gt = jnp.array(x[0], dtype=input_dtype[0]) | jnp.array(
        x[1], dtype=input_dtype[1]
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    for (u, v) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="jax",
        )


# __ror__
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("signed_integer"), num_arrays=2
    )
)
def test_jax_special_ror(
    dtype_x,
    fw,
):
    input_dtype, x = dtype_x
    ret = DeviceArray(x[1]).__ror__(DeviceArray(x[0]))
    ret_gt = jnp.array(x[1], dtype=input_dtype[1]).__ror__(
        jnp.array(x[0], dtype=input_dtype[0])
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    for (u, v) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="jax",
        )


# __xor__
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("signed_integer"), num_arrays=2
    )
)
def test_jax_special_xor(
    dtype_x,
    fw,
):
    input_dtype, x = dtype_x
    ret = DeviceArray(x[0]) ^ DeviceArray(x[1])
    ret_gt = jnp.array(x[0], dtype=input_dtype[0]) ^ jnp.array(
        x[1], dtype=input_dtype[1]
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    for (u, v) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="jax",
        )


# __rxor__
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("signed_integer"), num_arrays=2
    )
)
def test_jax_special_rxor(
    dtype_x,
    fw,
):
    input_dtype, x = dtype_x
    ret = DeviceArray(x[1]).__rxor__(DeviceArray(x[0]))
    ret_gt = jnp.array(x[1], dtype=input_dtype[1]).__rxor__(
        jnp.array(x[0], dtype=input_dtype[0])
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    for (u, v) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="jax",
        )


# __invert__
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("signed_integer")
    )
)
def test_jax_special_invert(
    dtype_x,
    fw,
):
    input_dtype, x = dtype_x
    ret = ~DeviceArray(x[0])
    ret_gt = ~jnp.array(x[0], dtype=input_dtype[0])
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    for (u, v) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="jax",
        )


# __lshift__
@handle_cmd_line_args
@given(dtype_x_shift=_get_dtype_x_and_int(dtype="signed_integer"))
def test_jax_special_lshift(
    dtype_x_shift,
    fw,
):
    input_dtype, x, shift = dtype_x_shift
    ret = DeviceArray(x[0]) << shift
    ret_gt = jnp.array(x[0], dtype=input_dtype[0]) << shift
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    for (u, v) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="jax",
        )


# __rlshift__
@handle_cmd_line_args
@given(dtype_x_shift=_get_dtype_x_and_int(dtype="signed_integer"))
def test_jax_special_rlshift(
    dtype_x_shift,
    fw,
):
    input_dtype, x, shift = dtype_x_shift
    ret = DeviceArray(shift).__rlshift__(DeviceArray(x[0]))
    ret_gt = jnp.array(shift).__rlshift__(jnp.array(x[0], dtype=input_dtype[0]))
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    for (u, v) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="jax",
        )


# __rshift__
@handle_cmd_line_args
@given(dtype_x_shift=_get_dtype_x_and_int(dtype="signed_integer"))
def test_jax_special_rshift(
    dtype_x_shift,
    fw,
):
    input_dtype, x, shift = dtype_x_shift
    ret = DeviceArray(x[0]) >> shift
    ret_gt = jnp.array(x[0], dtype=input_dtype[0]) >> shift
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    for (u, v) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="jax",
        )


# __rrshift__
@handle_cmd_line_args
@given(dtype_x_shift=_get_dtype_x_and_int(dtype="signed_integer"))
def test_jax_special_rrshift(
    dtype_x_shift,
    fw,
):
    input_dtype, x, shift = dtype_x_shift
    ret = DeviceArray(shift).__rrshift__(DeviceArray(x[0]))
    ret_gt = jnp.array(shift).__rrshift__(jnp.array(x[0], dtype=input_dtype[0]))
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    for (u, v) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="jax",
        )
