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
    fw,
):
    x, dtype, shape, dimensions = x_reshape_permut
    helpers.test_frontend_array_instance_method(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
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
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_array_instance_method(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        frontend_class=DeviceArray,
        fn_tree="DeviceArray.add",
        self=x[0],
        other=x[1],
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
        available_dtypes=helpers.get_dtypes("numeric", full=True),
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
    ret_gt = jnp.remainder(
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
        available_dtypes=helpers.get_dtypes("numeric", full=True),
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
        available_dtypes=helpers.get_dtypes("numeric", full=True),
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
    ret_gt = divmod(
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
        available_dtypes=helpers.get_dtypes("numeric", full=True),
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
    ret_gt = divmod(
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
        available_dtypes=helpers.get_dtypes("numeric", full=True),
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
        available_dtypes=helpers.get_dtypes("numeric", full=True),
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


# __matmul__
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=2,
        max_dim_size=10,
        shared_dtype=True,
        num_arrays=2,
    )
)
def test_jax_special_matmul(
    dtype_x,
    fw,
):
    input_dtype, x = dtype_x
    data = DeviceArray(x[0])
    other = DeviceArray(x[1])
    ret = data.__matmul__(other)
    ret_gt = jnp.matmul(
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


# __rmatmul__
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=2,
        max_dim_size=10,
        shared_dtype=True,
        num_arrays=2,
    )
)
def test_jax_special_rmatmul(
    dtype_x,
    fw,
):
    input_dtype, x = dtype_x
    data = DeviceArray(x[0])
    other = DeviceArray(x[1])
    ret = data.__rmatmul__(other)
    ret_gt = jnp.matmul(
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
