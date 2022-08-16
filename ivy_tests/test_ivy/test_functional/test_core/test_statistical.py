"""Collection of tests for statistical functions."""
# global
import numpy as np
from hypothesis import given, assume, settings, strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@st.composite
def statistical_dtype_values(draw, *, function):
    dtype = draw(st.sampled_from(ivy_np.valid_float_dtypes))
    size = draw(helpers.ints(min_value=1, max_value=10))
    if dtype == "float16":
        max_value = 2048
    elif dtype == "float32":
        max_value = 16777216
    elif dtype == "float64":
        max_value = 9.0071993e15

    if function == "prod":
        abs_value_limit = 0.99 * max_value ** (1 / size)
    elif function in ["var", "std"]:
        abs_value_limit = 0.99 * (max_value / size) ** 0.5
    else:
        abs_value_limit = 0.99 * max_value / size

    values = draw(
        helpers.list_of_length(
            x=helpers.floats(
                min_value=-abs_value_limit,
                max_value=abs_value_limit,
                allow_subnormal=False,
            ),
            length=size,
        )
    )
    shape = np.asarray(values, dtype=dtype).shape
    size = np.asarray(values, dtype=dtype).size
    axis = draw(helpers.get_axis(shape=shape, allow_none=True))
    if function == "var" or function == "std":
        if isinstance(axis, int):
            correction = draw(
                helpers.ints(min_value=-shape[axis], max_value=shape[axis] - 1)
                | helpers.floats(min_value=-shape[axis], max_value=shape[axis] - 1)
            )
            return dtype, values, axis, correction
        correction = draw(
            helpers.ints(min_value=-size, max_value=size - 1)
            | helpers.floats(min_value=-size, max_value=size - 1)
        )
        return dtype, values, axis, correction
    return dtype, values, axis


# min
@handle_cmd_line_args
@given(
    dtype_and_x=statistical_dtype_values(function="min"),
    num_positional_args=helpers.num_positional_args(fn_name="min"),
    container=st.booleans(),
    keep_dims=st.booleans(),
)
@settings(max_examples=1)
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
    container=st.booleans(),
    keep_dims=st.booleans(),
)
@settings(max_examples=1)
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
    container=st.booleans(),
    keep_dims=st.booleans(),
)
@settings(max_examples=1)
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
        x=np.asarray(x, dtype=input_dtype),
        axis=axis,
        keepdims=keep_dims,
    )


# var
@handle_cmd_line_args
@given(
    dtype_and_x=statistical_dtype_values(function="var"),
    num_positional_args=helpers.num_positional_args(fn_name="var"),
    container=st.booleans(),
    keep_dims=st.booleans(),
)
@settings(max_examples=1)
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
        x=np.asarray(x, dtype=input_dtype),
        axis=axis,
        correction=correction,
        keepdims=keep_dims,
    )


# prod
@handle_cmd_line_args
@given(
    dtype_and_x=statistical_dtype_values(function="prod"),
    num_positional_args=helpers.num_positional_args(fn_name="prod"),
    container=st.booleans(),
    keep_dims=st.booleans(),
)
@settings(max_examples=1)
def test_prod(
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
        fn_name="prod",
        x=np.asarray(x, dtype=input_dtype),
        axis=axis,
        keepdims=keep_dims,
        dtype=input_dtype,
    )


# sum
@handle_cmd_line_args
@given(
    dtype_and_x=statistical_dtype_values(function="sum"),
    num_positional_args=helpers.num_positional_args(fn_name="sum"),
    container=st.booleans(),
    keep_dims=st.booleans(),
)
@settings(max_examples=1)
def test_sum(
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
        fn_name="sum",
        rtol_=1e-2,
        x=np.asarray(x, dtype=input_dtype),
        axis=axis,
        keepdims=keep_dims,
        dtype=input_dtype,
    )


# std
@handle_cmd_line_args
@given(
    dtype_and_x=statistical_dtype_values(function="std"),
    num_positional_args=helpers.num_positional_args(fn_name="std"),
    container=st.booleans(),
    keep_dims=st.booleans(),
)
@settings(max_examples=1)
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
    dtype=st.sampled_from(ivy_np.valid_float_dtypes),
    with_out=st.booleans(),
    tensor_fn=st.sampled_from([ivy.array, helpers.var_fn]),
)
@settings(max_examples=1)
def test_einsum(*, eq_n_op_n_shp, dtype, with_out, tensor_fn, fw, device, call):
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
        call(ivy.einsum, eq, *operands),
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
