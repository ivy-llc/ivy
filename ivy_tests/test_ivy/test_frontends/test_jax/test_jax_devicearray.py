# global
from hypothesis import strategies as st
import jax.numpy as jnp

# local
from ivy.functional.frontends.jax.devicearray import DeviceArray
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
import ivy.functional.backends.torch as ivy_torch


# __pos__
@handle_frontend_test(
    fn_tree="jax.lax.add",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("numeric")),
)
def test_jax_special_pos(
    dtype_x,
):
    input_dtype, x = dtype_x
    ret = +DeviceArray(x[0])
    ret_gt = +jnp.array(x[0], dtype=input_dtype[0])
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend="jax",
    )


# __neg__
@handle_frontend_test(
    fn_tree="jax.lax.add",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("numeric")),
)
def test_jax_special_neg(
    dtype_x,
):
    input_dtype, x = dtype_x
    ret = -DeviceArray(x[0])
    ret_gt = -jnp.array(x[0], dtype=input_dtype[0])
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend="jax",
    )


# __eq__
@handle_frontend_test(
    fn_tree="jax.lax.add",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"), num_arrays=2
    ),
)
def test_jax_special_eq(
    dtype_x,
):
    input_dtype, x = dtype_x
    ret = DeviceArray(x[0]) == DeviceArray(x[1])
    ret_gt = jnp.array(x[0], dtype=input_dtype[0]) == jnp.array(
        x[1], dtype=input_dtype[1]
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend="jax",
    )


# __ne__
@handle_frontend_test(
    fn_tree="jax.lax.add",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"), num_arrays=2
    ),
)
def test_jax_special_ne(
    dtype_x,
):
    input_dtype, x = dtype_x
    ret = DeviceArray(x[0]) != DeviceArray(x[1])
    ret_gt = jnp.array(x[0], dtype=input_dtype[0]) != jnp.array(
        x[1], dtype=input_dtype[1]
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend="jax",
    )


# __lt__
@handle_frontend_test(
    fn_tree="jax.lax.add",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"), num_arrays=2
    ),
)
def test_jax_special_lt(
    dtype_x,
):
    input_dtype, x = dtype_x
    ret = DeviceArray(x[0]) < DeviceArray(x[1])
    ret_gt = jnp.array(x[0], dtype=input_dtype[0]) < jnp.array(
        x[1], dtype=input_dtype[1]
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend="jax",
    )


# __le__
@handle_frontend_test(
    fn_tree="jax.lax.add",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"), num_arrays=2
    ),
)
def test_jax_special_le(
    dtype_x,
):
    input_dtype, x = dtype_x
    ret = DeviceArray(x[0]) <= DeviceArray(x[1])
    ret_gt = jnp.array(x[0], dtype=input_dtype[0]) <= jnp.array(
        x[1], dtype=input_dtype[1]
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend="jax",
    )


# __gt__
@handle_frontend_test(
    fn_tree="jax.lax.add",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"), num_arrays=2
    ),
)
def test_jax_special_gt(
    dtype_x,
):
    input_dtype, x = dtype_x
    ret = DeviceArray(x[0]) > DeviceArray(x[1])
    ret_gt = jnp.array(x[0], dtype=input_dtype[0]) > jnp.array(
        x[1], dtype=input_dtype[1]
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend="jax",
    )


# __ge__
@handle_frontend_test(
    fn_tree="jax.lax.add",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"), num_arrays=2
    ),
)
def test_jax_special_ge(
    dtype_x,
):
    input_dtype, x = dtype_x
    ret = DeviceArray(x[0]) >= DeviceArray(x[1])
    ret_gt = jnp.array(x[0], dtype=input_dtype[0]) >= jnp.array(
        x[1], dtype=input_dtype[1]
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend="jax",
    )


# __abs__
@handle_frontend_test(
    fn_tree="jax.lax.add",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("numeric")),
)
def test_jax_special_abs(
    dtype_x,
):
    input_dtype, x = dtype_x
    ret = abs(DeviceArray(x[0]))
    ret_gt = abs(jnp.array(x[0], dtype=input_dtype[0]))
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
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
@handle_frontend_test(
    fn_tree="jax.lax.add", dtype_x_pow=_get_dtype_x_and_int()  # dummy fn_tree
)
def test_jax_special_pow(
    dtype_x_pow,
):
    x_dtype, x, pow = dtype_x_pow
    ret = DeviceArray(x[0]) ** pow
    ret_gt = jnp.array(x[0], dtype=x_dtype[0]) ** pow
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend="jax",
    )


# __rpow__
@handle_frontend_test(
    fn_tree="jax.lax.add", dtype_x_pow=_get_dtype_x_and_int()  # dummy fn_tree
)
def test_jax_special_rpow(
    dtype_x_pow,
):
    x_dtype, x, pow = dtype_x_pow
    ret = DeviceArray(pow).__rpow__(DeviceArray(x[0]))
    ret_gt = jnp.array(pow).__rpow__(jnp.array(x[0], dtype=x_dtype[0]))
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend="jax",
    )


# __and__
@handle_frontend_test(
    fn_tree="jax.lax.add",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("signed_integer"), num_arrays=2
    ),
)
def test_jax_special_and(
    dtype_x,
):
    input_dtype, x = dtype_x
    ret = DeviceArray(x[0]) & DeviceArray(x[1])
    ret_gt = jnp.array(x[0], dtype=input_dtype[0]) & jnp.array(
        x[1], dtype=input_dtype[1]
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend="jax",
    )


# __rand__
@handle_frontend_test(
    fn_tree="jax.lax.add",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("signed_integer"), num_arrays=2
    ),
)
def test_jax_special_rand(
    dtype_x,
):
    input_dtype, x = dtype_x
    ret = DeviceArray(x[1]).__rand__(DeviceArray(x[0]))
    ret_gt = jnp.array(x[1], dtype=input_dtype[1]).__rand__(
        jnp.array(x[0], dtype=input_dtype[0])
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend="jax",
    )


# __or__
@handle_frontend_test(
    fn_tree="jax.lax.add",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("signed_integer"), num_arrays=2
    ),
)
def test_jax_special_or(
    dtype_x,
):
    input_dtype, x = dtype_x
    ret = DeviceArray(x[0]) | DeviceArray(x[1])
    ret_gt = jnp.array(x[0], dtype=input_dtype[0]) | jnp.array(
        x[1], dtype=input_dtype[1]
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend="jax",
    )


# __ror__
@handle_frontend_test(
    fn_tree="jax.lax.add",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("signed_integer"), num_arrays=2
    ),
)
def test_jax_special_ror(
    dtype_x,
):
    input_dtype, x = dtype_x
    ret = DeviceArray(x[1]).__ror__(DeviceArray(x[0]))
    ret_gt = jnp.array(x[1], dtype=input_dtype[1]).__ror__(
        jnp.array(x[0], dtype=input_dtype[0])
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend="jax",
    )


# __xor__
@handle_frontend_test(
    fn_tree="jax.lax.add",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("signed_integer"), num_arrays=2
    ),
)
def test_jax_special_xor(
    dtype_x,
):
    input_dtype, x = dtype_x
    ret = DeviceArray(x[0]) ^ DeviceArray(x[1])
    ret_gt = jnp.array(x[0], dtype=input_dtype[0]) ^ jnp.array(
        x[1], dtype=input_dtype[1]
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend="jax",
    )


# __rxor__
@handle_frontend_test(
    fn_tree="jax.lax.add",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("signed_integer"), num_arrays=2
    ),
)
def test_jax_special_rxor(
    dtype_x,
):
    input_dtype, x = dtype_x
    ret = DeviceArray(x[1]).__rxor__(DeviceArray(x[0]))
    ret_gt = jnp.array(x[1], dtype=input_dtype[1]).__rxor__(
        jnp.array(x[0], dtype=input_dtype[0])
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend="jax",
    )


# __invert__
@handle_frontend_test(
    fn_tree="jax.lax.add",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("signed_integer")
    ),
)
def test_jax_special_invert(
    dtype_x,
):
    input_dtype, x = dtype_x
    ret = ~DeviceArray(x[0])
    ret_gt = ~jnp.array(x[0], dtype=input_dtype[0])
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend="jax",
    )


# __lshift__
@handle_frontend_test(
    fn_tree="jax.lax.add",  # dummy fn_tree
    dtype_x_shift=_get_dtype_x_and_int(dtype="signed_integer"),
)
def test_jax_special_lshift(
    dtype_x_shift,
):
    input_dtype, x, shift = dtype_x_shift
    ret = DeviceArray(x[0]) << shift
    ret_gt = jnp.array(x[0], dtype=input_dtype[0]) << shift
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend="jax",
    )


# __rlshift__
@handle_frontend_test(
    fn_tree="jax.lax.add",  # dummy fn_tree
    dtype_x_shift=_get_dtype_x_and_int(dtype="signed_integer"),
)
def test_jax_special_rlshift(
    dtype_x_shift,
):
    input_dtype, x, shift = dtype_x_shift
    ret = DeviceArray(shift).__rlshift__(DeviceArray(x[0]))
    ret_gt = jnp.array(shift).__rlshift__(jnp.array(x[0], dtype=input_dtype[0]))
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend="jax",
    )


# __rshift__
@handle_frontend_test(
    fn_tree="jax.lax.add",  # dummy fn_tree
    dtype_x_shift=_get_dtype_x_and_int(dtype="signed_integer"),
)
def test_jax_special_rshift(
    dtype_x_shift,
):
    input_dtype, x, shift = dtype_x_shift
    ret = DeviceArray(x[0]) >> shift
    ret_gt = jnp.array(x[0], dtype=input_dtype[0]) >> shift
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend="jax",
    )


# __rrshift__
@handle_frontend_test(
    fn_tree="jax.lax.add",  # dummy fn_tree
    dtype_x_shift=_get_dtype_x_and_int(dtype="signed_integer"),
)
def test_jax_special_rrshift(
    dtype_x_shift,
):
    input_dtype, x, shift = dtype_x_shift
    ret = DeviceArray(shift).__rrshift__(DeviceArray(x[0]))
    ret_gt = jnp.array(shift).__rrshift__(jnp.array(x[0], dtype=input_dtype[0]))
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend="jax",
    )


# __add__
@handle_frontend_test(
    fn_tree="jax.lax.add",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
        shared_dtype=True,
        num_arrays=2,
    ),
)
def test_jax_special_add(
    dtype_x,
):
    input_dtype, x = dtype_x
    ret = DeviceArray(x[0]) + DeviceArray(x[1])
    ret_gt = jnp.array(x[0]) + jnp.array(x[1], dtype=input_dtype[1])
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend="jax",
    )


# __radd__
@handle_frontend_test(
    fn_tree="jax.lax.add",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
        shared_dtype=True,
        num_arrays=2,
    ),
)
def test_jax_special_radd(
    dtype_x,
):
    input_dtype, x = dtype_x
    data = DeviceArray(x[0])
    other = DeviceArray(x[1])
    ret = data.__radd__(other)
    ret_gt = jnp.array(x[0], dtype=input_dtype[0]).__radd__(
        jnp.array(x[1], dtype=input_dtype[1])
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend="jax",
    )


# __sub__
@handle_frontend_test(
    fn_tree="jax.lax.add",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
        shared_dtype=True,
        num_arrays=2,
    ),
)
def test_jax_special_sub(
    dtype_x,
):
    input_dtype, x = dtype_x
    ret = DeviceArray(x[0]) - DeviceArray(x[1])
    ret_gt = jnp.array(x[0], dtype=input_dtype[0]) - jnp.array(
        x[1], dtype=input_dtype[1]
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend="jax",
    )


# __rsub__
@handle_frontend_test(
    fn_tree="jax.lax.add",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
        shared_dtype=True,
        num_arrays=2,
    ),
)
def test_jax_special_rsub(
    dtype_x,
):
    input_dtype, x = dtype_x
    data = DeviceArray(x[0])
    other = DeviceArray(x[1])
    ret = data.__rsub__(other)
    ret_gt = jnp.array(x[0], dtype=input_dtype[0]).__rsub__(
        jnp.array(x[1], dtype=input_dtype[1])
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend="jax",
    )


# __mul__
@handle_frontend_test(
    fn_tree="jax.lax.add",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
        shared_dtype=True,
        num_arrays=2,
    ),
)
def test_jax_special_mul(
    dtype_x,
):
    input_dtype, x = dtype_x
    ret = DeviceArray(x[0]) * DeviceArray(x[1])
    ret_gt = jnp.array(x[0], dtype=input_dtype[0]) * jnp.array(
        x[1], dtype=input_dtype[1]
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend="jax",
    )


# __rmul__
@handle_frontend_test(
    fn_tree="jax.lax.add",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
        shared_dtype=True,
        num_arrays=2,
    ),
)
def test_jax_special_rmul(
    dtype_x,
):
    input_dtype, x = dtype_x
    data = DeviceArray(x[0])
    other = DeviceArray(x[1])
    ret = data.__rmul__(other)
    ret_gt = jnp.array(x[0], dtype=input_dtype[0]).__rmul__(
        jnp.array(x[1], dtype=input_dtype[1])
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend="jax",
    )


# __div__
@handle_frontend_test(
    fn_tree="jax.lax.add",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
        shared_dtype=True,
        num_arrays=2,
    ),
)
def test_jax_special_div(
    dtype_x,
):
    input_dtype, x = dtype_x
    ret = DeviceArray(x[0]) / DeviceArray(x[1])
    ret_gt = jnp.array(x[0], dtype=input_dtype[0]) / jnp.array(
        x[1], dtype=input_dtype[1]
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend="jax",
    )


# __rdiv__
@handle_frontend_test(
    fn_tree="jax.lax.add",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
        shared_dtype=True,
        num_arrays=2,
    ),
)
def test_jax_special_rdiv(
    dtype_x,
):
    input_dtype, x = dtype_x
    data = DeviceArray(x[0])
    other = DeviceArray(x[1])
    ret = data.__rdiv__(other)
    ret_gt = jnp.array(x[0], dtype=input_dtype[0]).__rdiv__(
        jnp.array(x[1], dtype=input_dtype[1])
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend="jax",
    )


# __truediv__
@handle_frontend_test(
    fn_tree="jax.lax.add",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
        shared_dtype=True,
        num_arrays=2,
    ),
)
def test_jax_special_truediv(
    dtype_x,
):
    input_dtype, x = dtype_x
    data = DeviceArray(x[0])
    other = DeviceArray(x[1])
    ret = data.__truediv__(other)
    ret_gt = jnp.array(x[0], dtype=input_dtype[0]).__truediv__(
        jnp.array(x[1], dtype=input_dtype[1])
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend="jax",
    )


# __rtruediv__
@handle_frontend_test(
    fn_tree="jax.lax.add",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
        shared_dtype=True,
        num_arrays=2,
    ),
)
def test_jax_special_rtruediv(
    dtype_x,
):
    input_dtype, x = dtype_x
    data = DeviceArray(x[0])
    other = DeviceArray(x[1])
    ret = data.__rtruediv__(other)
    ret_gt = jnp.array(x[0]).__rtruediv__(jnp.array(x[1], dtype=input_dtype[1]))
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend="jax",
    )


# __mod__
@handle_frontend_test(
    fn_tree="jax.lax.add",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", full=True),
        shared_dtype=True,
        num_arrays=2,
    ),
)
def test_jax_special_mod(
    dtype_x,
):
    input_dtype, x = dtype_x
    data = DeviceArray(x[0])
    other = DeviceArray(x[1])
    ret = data % other
    ret_gt = jnp.array(x[0], dtype=input_dtype[0]) % jnp.array(
        x[1], dtype=input_dtype[1]
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend="jax",
    )


# __rmod__
@handle_frontend_test(
    fn_tree="jax.lax.add",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", full=True),
        shared_dtype=True,
        num_arrays=2,
    ),
)
def test_jax_special_rmod(
    dtype_x,
):
    input_dtype, x = dtype_x
    data = DeviceArray(x[0])
    other = DeviceArray(x[1])
    ret = data.__rmod__(other)
    ret_gt = jnp.array(x[0], dtype=input_dtype[0]).__rmod__(
        jnp.array(x[1], dtype=input_dtype[1])
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
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
@handle_frontend_test(
    fn_tree="jax.lax.add",  # dummy fn_tree
    dtype_x=_get_dtype_input_and_vectors(),
)
def test_jax_special_matmul(
    dtype_x,
):
    input_dtype, x = dtype_x
    data = DeviceArray(x[0])
    other = DeviceArray(x[1])
    ret = data @ other
    ret_gt = jnp.array(x[0]) @ jnp.array(x[1])
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend="jax",
    )


# __rmatmul__
@handle_frontend_test(
    fn_tree="jax.lax.add",  # dummy fn_tree
    dtype_x=_get_dtype_input_and_vectors(),
)
def test_jax_special_rmatmul(
    dtype_x,
):
    input_dtype, x = dtype_x
    data = DeviceArray(x[0])
    other = DeviceArray(x[1])
    ret = data.__rmatmul__(other)
    ret_gt = jnp.array(x[1]).__rmatmul__(jnp.array(x[0]))
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend="jax",
    )


# __getitem__
@handle_frontend_test(
    fn_tree="jax.lax.add",  # dummy fn_tree
    dtype_x_index=helpers.array_indices_axis(
        array_dtypes=helpers.get_dtypes("numeric"),
        indices_dtypes=ivy_torch.valid_int_dtypes,
    ),
)
def test_jax_special_getitem(
    dtype_x_index,
):
    x, index = dtype_x_index[1:3]
    ret = DeviceArray(x).__getitem__(index)
    ret_gt = jnp.array(x).at[index].get()
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend="jax",
    )
