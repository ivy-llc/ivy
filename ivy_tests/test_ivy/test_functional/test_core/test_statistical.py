"""Collection of tests for statistical functions."""
# global
import numpy as np
from hypothesis import given, assume, strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@st.composite
def statistical_dtype_values(draw, *, function):
    large_abs_safety_factor = 2
    small_abs_safety_factor = 2
    if function in ["mean", "std", "var"]:
        large_abs_safety_factor = 24
        small_abs_safety_factor = 24
    dtype, values, axis = draw(
        helpers.dtype_values_axis(
            available_dtypes=helpers.get_dtypes("float"),
            large_abs_safety_factor=large_abs_safety_factor,
            small_abs_safety_factor=small_abs_safety_factor,
            safety_factor_scale="log",
            min_num_dims=1,
            max_num_dims=5,
            min_dim_size=2,
            valid_axis=True,
            allow_neg_axes=False,
            min_axes_size=1,
        )
    )
    shape = values[0].shape
    size = values[0].size
    max_correction = np.min(shape)
    if function == "var" or function == "std":
        if size == 1:
            correction = 0
        elif isinstance(axis, int):
            correction = draw(
                helpers.ints(min_value=0, max_value=shape[axis] - 1)
                | helpers.floats(min_value=0, max_value=shape[axis] - 1)
            )
            return dtype, values, axis, correction
        else:
            correction = draw(
                helpers.ints(min_value=0, max_value=max_correction - 1)
                | helpers.floats(min_value=0, max_value=max_correction - 1)
            )
        return dtype, values, axis, correction
    return dtype, values, axis


# min
@handle_cmd_line_args
@given(
    dtype_and_x=statistical_dtype_values(function="min"),
    num_positional_args=helpers.num_positional_args(fn_name="min"),
    keep_dims=st.booleans(),
)
def test_min(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    keep_dims,
):
    input_dtype, x, axis = dtype_and_x
    assume(x)
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="min",
        x=np.asarray(x, dtype=input_dtype),
        axis=axis,
        keepdims=keep_dims,
    )


# max
@handle_cmd_line_args
@given(
    dtype_and_x=statistical_dtype_values(function="max"),
    num_positional_args=helpers.num_positional_args(fn_name="max"),
    keep_dims=st.booleans(),
)
def test_max(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    keep_dims,
):
    input_dtype, x, axis = dtype_and_x
    assume(x)
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="max",
        x=np.asarray(x, dtype=input_dtype),
        axis=axis,
        keepdims=keep_dims,
    )


# mean
@handle_cmd_line_args
@given(
    dtype_and_x=statistical_dtype_values(function="mean"),
    num_positional_args=helpers.num_positional_args(fn_name="mean"),
    keep_dims=st.booleans(),
)
def test_mean(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    keep_dims,
):
    input_dtype, x, axis = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="mean",
        rtol_=1e-1,
        atol_=1e-1,
        x=np.asarray(x, dtype=input_dtype),
        axis=axis,
        keepdims=keep_dims,
    )


# var
@handle_cmd_line_args
@given(
    dtype_and_x=statistical_dtype_values(function="var"),
    num_positional_args=helpers.num_positional_args(fn_name="var"),
    keep_dims=st.booleans(),
)
def test_var(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    keep_dims,
):
    input_dtype, x, axis, correction = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="var",
        rtol_=1e-1,
        atol_=1e-2,
        x=np.asarray(x, dtype=input_dtype),
        axis=axis,
        correction=correction,
        keepdims=keep_dims,
    )


# prod
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
    num_positional_args=helpers.num_positional_args(fn_name="prod"),
    keep_dims=st.booleans(),
    dtype=helpers.get_dtypes("numeric", none=True),
)
def test_prod(
    *,
    dtype_x_axis,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    keep_dims,
    dtype,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="prod",
        x=np.asarray(x, dtype=input_dtype),
        axis=axis,
        keepdims=keep_dims,
        dtype=dtype,
    )


# sum
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
    num_positional_args=helpers.num_positional_args(fn_name="sum"),
    keep_dims=st.booleans(),
    dtype=helpers.get_dtypes("numeric", none=True),
)
def test_sum(
    *,
    dtype_x_axis,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    keep_dims,
    dtype,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="sum",
        rtol_=1e-1,
        atol_=1e-2,
        x=np.asarray(x, dtype=input_dtype),
        axis=axis,
        keepdims=keep_dims,
        dtype=dtype,
    )


# std
@handle_cmd_line_args
@given(
    dtype_and_x=statistical_dtype_values(function="std"),
    num_positional_args=helpers.num_positional_args(fn_name="std"),
    keep_dims=st.booleans(),
)
def test_std(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    keep_dims,
):
    input_dtype, x, axis, correction = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="std",
        rtol_=1e-2,
        atol_=1e-2,
        x=np.asarray(x, dtype=input_dtype),
        axis=axis,
        correction=correction,
        keepdims=keep_dims,
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
    num_positional_args=helpers.num_positional_args(fn_name="cumsum"),
    exclusive=st.booleans(),
    reverse=st.booleans(),
    dtype=helpers.get_dtypes("numeric", none=True),
)
def test_cumsum(
    dtype_x_axis,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    exclusive,
    reverse,
    dtype,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="cumsum",
        x=np.asarray(x, dtype=input_dtype),
        axis=axis,
        exclusive=exclusive,
        reverse=reverse,
        dtype=dtype,
    )


# cumprod
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
    num_positional_args=helpers.num_positional_args(fn_name="cumprod"),
    exclusive=st.booleans(),
    dtype=helpers.get_dtypes("numeric", none=True),
)
def test_cumprod(
    dtype_x_axis,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    exclusive,
    dtype,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="cumprod",
        x=np.asarray(x, dtype=input_dtype),
        axis=axis,
        exclusive=exclusive,
        dtype=dtype,
    )


# einsum
@handle_cmd_line_args
@given(
    eq_n_op_n_shp=st.sampled_from(
        [
            ("ii", (np.arange(25).reshape(5, 5),), ()),
            ("ii->i", (np.arange(25).reshape(5, 5),), (5,)),
            ("ij,j", (np.arange(25).reshape(5, 5), np.arange(5)), (5,)),
        ]
    ),
    dtype=helpers.get_dtypes("float", full=False),
    with_out=st.booleans(),
    tensor_fn=st.sampled_from([ivy.array, helpers.var_fn]),
)
def test_einsum(*, eq_n_op_n_shp, dtype, with_out, tensor_fn, fw, device):
    # bfloat16 is not supported by numpy
    assume(not ("bfloat16" in dtype))
    # smoke test
    eq, operands, true_shape = eq_n_op_n_shp
    operands = [tensor_fn(op, dtype=dtype, device=device) for op in operands]
    if with_out:
        out = ivy.zeros(true_shape, dtype=dtype)
        ret = ivy.einsum(eq, *operands, out=out)
    else:
        ret = ivy.einsum(eq, *operands)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == true_shape
    # value test
    assert np.allclose(
        ivy.to_numpy(ivy.einsum(eq, *operands)),
        ivy.functional.backends.numpy.einsum(
            eq, *[ivy.to_numpy(op) for op in operands]
        ),
    )
    # out test
    if with_out:
        assert ret is out

        # these backends do not support native inplace updates
        assume(not (fw in ["tensorflow", "jax"]))

        assert ret.data is out.data
