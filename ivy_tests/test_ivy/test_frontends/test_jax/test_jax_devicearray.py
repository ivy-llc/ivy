# global
from hypothesis import given, strategies as st

# local
import jax.numpy as jnp
from ivy.functional.frontends.jax.devicearray import DeviceArray
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


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


# __add__
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
        shared_dtype=True,
        num_arrays=2,
    )
)
def test_jax_special_add(
    dtype_x,
    fw,
):
    input_dtype, x = dtype_x
    ret = DeviceArray(x[0]) + DeviceArray(x[1])
    ret_gt = jnp.array(x[0], dtype=input_dtype[0]) + jnp.array(
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


# __radd__
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
        shared_dtype=True,
        num_arrays=2,
    )
)
def test_jax_special_radd(
    dtype_x,
    fw,
):
    input_dtype, x = dtype_x
    data = DeviceArray(x[0])
    other = DeviceArray(x[1])
    ret = data.__radd__(other)
    ret_gt = jnp.array(x[0], dtype=input_dtype[0]) + jnp.array(
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


# __sub__
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
        shared_dtype=True,
        num_arrays=2,
    )
)
def test_jax_special_sub(
    dtype_x,
    fw,
):
    input_dtype, x = dtype_x
    ret = DeviceArray(x[0]) - DeviceArray(x[1])
    ret_gt = jnp.array(x[0], dtype=input_dtype[0]) - jnp.array(
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


# __rsub__
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
        shared_dtype=True,
        num_arrays=2,
    )
)
def test_jax_special_rsub(
    dtype_x,
    fw,
):
    input_dtype, x = dtype_x
    data = DeviceArray(x[0])
    other = DeviceArray(x[1])
    ret = data.__rsub__(other)
    ret_gt = jnp.array(x[0], dtype=input_dtype[0]) - jnp.array(
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


# __mul__
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
        shared_dtype=True,
        num_arrays=2,
    )
)
def test_jax_special_mul(
    dtype_x,
    fw,
):
    input_dtype, x = dtype_x
    ret = DeviceArray(x[0]) * DeviceArray(x[1])
    ret_gt = jnp.array(x[0], dtype=input_dtype[0]) * jnp.array(
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


# __rmul__
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
        shared_dtype=True,
        num_arrays=2,
    )
)
def test_jax_special_rmul(
    dtype_x,
    fw,
):
    input_dtype, x = dtype_x
    data = DeviceArray(x[0])
    other = DeviceArray(x[1])
    ret = data.__rmul__(other)
    ret_gt = jnp.array(x[0], dtype=input_dtype[0]) * jnp.array(
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


# __div__
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
        shared_dtype=True,
        num_arrays=2,
    )
)
def test_jax_special_div(
    dtype_x,
    fw,
):
    input_dtype, x = dtype_x
    ret = DeviceArray(x[0]) / DeviceArray(x[1])
    ret_gt = jnp.array(x[0], dtype=input_dtype[0]) / jnp.array(
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


# __rdiv__
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
        shared_dtype=True,
        num_arrays=2,
    )
)
def test_jax_special_rdiv(
    dtype_x,
    fw,
):
    input_dtype, x = dtype_x
    data = DeviceArray(x[0])
    other = DeviceArray(x[1])
    ret = data.__rdiv__(other)
    ret_gt = jnp.array(x[1], dtype=input_dtype[1]) / jnp.array(
        x[0], dtype=input_dtype[0]
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    for (u, v) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="jax",
        )


# __truediv__
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
        shared_dtype=True,
        num_arrays=2,
    )
)
def test_jax_special_truediv(
    dtype_x,
    fw,
):
    input_dtype, x = dtype_x
    data = DeviceArray(x[0])
    other = DeviceArray(x[1])
    ret = data.__truediv__(other)
    ret_gt = jnp.array(x[0], dtype=input_dtype[0]) / jnp.array(
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


# __rtruediv__
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
        shared_dtype=True,
        num_arrays=2,
    )
)
def test_jax_special_rtruediv(
    dtype_x,
    fw,
):
    input_dtype, x = dtype_x
    data = DeviceArray(x[0])
    other = DeviceArray(x[1])
    ret = data.__rtruediv__(other)
    ret_gt = jnp.array(x[1], dtype=input_dtype[1]) / jnp.array(
        x[0], dtype=input_dtype[0]
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    for (u, v) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="jax",
        )


# __mod__
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", full=True),
        shared_dtype=True,
        num_arrays=2,
    )
)
def test_jax_special_mod(
    dtype_x,
    fw,
):
    input_dtype, x = dtype_x
    data = DeviceArray(x[0])
    other = DeviceArray(x[1])
    ret = data.__mod__(other)
    ret_gt = jnp.mod(
        jnp.array(x[0], dtype=input_dtype[0]), jnp.array(x[1], dtype=input_dtype[1])
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    for (u, v) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="jax",
        )


# __rmod__
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", full=True),
        shared_dtype=True,
        num_arrays=2,
    )
)
def test_jax_special_rmod(
    dtype_x,
    fw,
):
    input_dtype, x = dtype_x
    data = DeviceArray(x[0])
    other = DeviceArray(x[1])
    ret = data.__rmod__(other)
    ret_gt = jnp.remainder(
        jnp.array(x[1], dtype=input_dtype[1]), jnp.array(x[0], dtype=input_dtype[0])
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    for (u, v) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="jax",
        )


# __divmod__
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", full=True),
        shared_dtype=True,
        num_arrays=2,
    )
)
def test_jax_special_divmod(
    dtype_x,
    fw,
):
    input_dtype, x = dtype_x
    data = DeviceArray(x[0])
    other = DeviceArray(x[1])
    ret = data.__divmod__(other)
    ret_gt = jnp.divmod(
        jnp.array(x[0], dtype=input_dtype[0]), jnp.array(x[1], dtype=input_dtype[1])
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    for (u, v) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="jax",
        )


# __rdivmod__
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", full=True),
        shared_dtype=True,
        num_arrays=2,
    )
)
def test_jax_special_rdivmod(
    dtype_x,
    fw,
):
    input_dtype, x = dtype_x
    data = DeviceArray(x[0])
    other = DeviceArray(x[1])
    ret = data.__rdivmod__(other)
    ret_gt = jnp.divmod(
        jnp.array(x[1], dtype=input_dtype[1]), jnp.array(x[0], dtype=input_dtype[0])
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    for (u, v) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="jax",
        )


# __floordiv__
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", full=True),
        shared_dtype=True,
        num_arrays=2,
    )
)
def test_jax_special_floordiv(
    dtype_x,
    fw,
):
    input_dtype, x = dtype_x
    data = DeviceArray(x[0])
    other = DeviceArray(x[1])
    ret = data.__floordiv__(other)
    ret_gt = jnp.floor_divide(
        jnp.array(x[0], dtype=input_dtype[0]), jnp.array(x[1], dtype=input_dtype[1])
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    for (u, v) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="jax",
        )


# __rfloordiv__
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        exclude_min=True,
        shared_dtype=True,
        num_arrays=2,
    )
)
def test_jax_special_rfloordiv(
    dtype_x,
    fw,
):
    input_dtype, x = dtype_x
    data = DeviceArray(x[0])
    other = DeviceArray(x[1])
    ret = data.__rfloordiv__(other)
    ret_gt = jnp.floor_divide(
        jnp.array(x[1], dtype=input_dtype[1]), jnp.array(x[0], dtype=input_dtype[0])
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    for (u, v) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="jax",
        )


@st.composite
def _get_dtype_input_and_vectors(draw):
    dim_size = draw(helpers.ints(min_value=2, max_value=5))
    dtype = draw(helpers.get_dtypes("numeric", index=1, full=False))
    vec1 = draw(
        helpers.array_values(
            dtype=dtype[0], shape=(dim_size, dim_size), min_value=2, max_value=5
        )
    )
    vec2 = draw(
        helpers.array_values(
            dtype=dtype[0], shape=(dim_size, dim_size), min_value=2, max_value=5
        )
    )
    return dtype, [vec1, vec2]


# __matmul__
@handle_cmd_line_args
@given(
    dtype_x=_get_dtype_input_and_vectors(),
)
def test_jax_special_matmul(
    dtype_x,
    fw,
):
    input_dtype, x = dtype_x
    data = DeviceArray(x[0])
    other = DeviceArray(x[1])
    ret = data.__matmul__(other)
    ret_gt = jnp.matmul(jnp.array(x[0]), jnp.array(x[1]))
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    for (u, v) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="jax",
        )


# __rmatmul__
@handle_cmd_line_args
@given(
    dtype_x=_get_dtype_input_and_vectors(),
)
def test_jax_special_rmatmul(
    dtype_x,
    fw,
):
    input_dtype, x = dtype_x
    data = DeviceArray(x[0])
    other = DeviceArray(x[1])
    ret = data.__rmatmul__(other)
    ret_gt = jnp.matmul(jnp.array(x[1]), jnp.array(x[0]))
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    for (u, v) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="jax",
        )
