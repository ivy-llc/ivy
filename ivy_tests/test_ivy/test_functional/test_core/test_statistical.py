"""Collection of tests for statistical functions."""
# global
import numpy as np
from hypothesis import given, assume, strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@st.composite
def statistical_dtype_values(draw, *, function):
    dtype = draw(st.sampled_from(ivy_np.valid_float_dtypes))
    size = draw(st.integers(1, 10))
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
            x=st.floats(
                -abs_value_limit,
                abs_value_limit,
                allow_subnormal=False,
                allow_infinity=False,
            ),
            length=size,
        )
    )
    shape = np.asarray(values, dtype=dtype).shape
    axis = draw(helpers.get_axis(shape=shape, allow_none=True))
    return dtype, values, axis


@st.composite
def statistical_correction_values(draw, *, function):
    correction = draw(
        st.integers()
        | st.floats()
    )
    return correction


# min
@given(
    dtype_and_x=statistical_dtype_values(function="min"),
    num_positional_args=helpers.num_positional_args(fn_name="min"),
    data=st.data(),
)
@handle_cmd_line_args
def test_min(
    *,
    dtype_and_x,
    num_positional_args,
    data,
    fw,
):
    input_dtype, x, axis = dtype_and_x
    as_variable = data.draw(st.booleans())
    with_out = data.draw(st.booleans())
    native_array = data.draw(st.booleans())
    container = data.draw(st.booleans())
    instance_method = data.draw(st.booleans())
    keep_dims = data.draw(st.booleans())
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
        axis=None,  # axis=axis,
        keepdims=keep_dims,
    )


# max
@given(
    dtype_and_x=statistical_dtype_values(function="max"),
    num_positional_args=helpers.num_positional_args(fn_name="max"),
    data=st.data(),
)
@handle_cmd_line_args
def test_max(
    *,
    dtype_and_x,
    num_positional_args,
    data,
    fw,
):
    input_dtype, x, axis = dtype_and_x
    as_variable = data.draw(st.booleans())
    with_out = data.draw(st.booleans())
    native_array = data.draw(st.booleans())
    container = data.draw(st.booleans())
    instance_method = data.draw(st.booleans())
    keep_dims = data.draw(st.booleans())
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
        axis=None,  # axis=axis,
        keepdims=keep_dims,
    )


# mean
@given(
    dtype_and_x=statistical_dtype_values(function="mean"),
    num_positional_args=helpers.num_positional_args(fn_name="mean"),
    data=st.data(),
)
@handle_cmd_line_args
def test_mean(
    *,
    dtype_and_x,
    num_positional_args,
    data,
    fw,
):
    input_dtype, x, axis = dtype_and_x
    as_variable = data.draw(st.booleans())
    with_out = data.draw(st.booleans())
    native_array = data.draw(st.booleans())
    container = data.draw(st.booleans())
    instance_method = data.draw(st.booleans())
    keep_dims = data.draw(st.booleans())
    if fw == "torch" and (input_dtype in ivy_np.valid_int_dtypes):
        return  # torch implementation exhibits strange behaviour
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
        axis=None,  # axis=axis,
        keepdims=keep_dims
    )


# var
@given(
    dtype_and_x=statistical_dtype_values(function="var"),
    num_positional_args=helpers.num_positional_args(fn_name="var"),
    data=st.data(),
)
@handle_cmd_line_args
def test_var(
    *,
    dtype_and_x,
    num_positional_args,
    data,
    fw,
):
    input_dtype, x, axis = dtype_and_x
    as_variable = data.draw(st.booleans())
    with_out = data.draw(st.booleans())
    native_array = data.draw(st.booleans())
    container = data.draw(st.booleans())
    instance_method = data.draw(st.booleans())
    correction = data.draw(statistical_correction_values(function="std"))
    keep_dims = data.draw(st.booleans())
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
        axis=None,  # axis=axis,
        correction=correction,
        keepdims=keep_dims,
    )


# prod
@given(
    dtype_and_x=statistical_dtype_values(function="prod"),
    num_positional_args=helpers.num_positional_args(fn_name="prod"),
    data=st.data(),
)
@handle_cmd_line_args
def test_prod(
    *,
    dtype_and_x,
    num_positional_args,
    data,
    fw,
):
    input_dtype, x, axis = dtype_and_x
    as_variable = data.draw(st.booleans())
    with_out = data.draw(st.booleans())
    native_array = data.draw(st.booleans())
    container = data.draw(st.booleans())
    instance_method = data.draw(st.booleans())
    keep_dims=data.draw(st.booleans())
    if fw == "torch" and (input_dtype == "float16" or ivy.is_int_dtype(input_dtype)):
        return  # torch implementation exhibits strange behaviour
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
        axis=None,  # axis=axis,
        keepdims=keep_dims,
    )


# sum
@given(
    dtype_and_x=statistical_dtype_values(function="sum"),
    num_positional_args=helpers.num_positional_args(fn_name="sum"),
    data=st.data(),
)
@handle_cmd_line_args
def test_sum(
    *,
    dtype_and_x,
    num_positional_args,
    data,
    fw,
):
    input_dtype, x, axis = dtype_and_x
    as_variable = data.draw(st.booleans())
    with_out = data.draw(st.booleans())
    native_array = data.draw(st.booleans())
    container = data.draw(st.booleans())
    instance_method = data.draw(st.booleans())
    keep_dims=data.draw(st.booleans())
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
        axis=None,  # axis=axis,
        keepdims=keep_dims,
    )


# std
@given(
    dtype_and_x=statistical_dtype_values(function="std"),
    num_positional_args=helpers.num_positional_args(fn_name="std"),
    data=st.data(),
)
@handle_cmd_line_args
def test_std(
    *,
    dtype_and_x,
    num_positional_args,
    data,
    fw,
):
    input_dtype, x, axis = dtype_and_x
    as_variable = data.draw(st.booleans())
    with_out = data.draw(st.booleans())
    native_array = data.draw(st.booleans())
    container = data.draw(st.booleans())
    instance_method = data.draw(st.booleans())
    correction=data.draw(statistical_correction_values(function="std"))
    keep_dims=data.draw(st.booleans())
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
        axis=None,  # axis=axis,
        correction=correction,
        keepdims=keep_dims,
    )


# einsum
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
    data=st.data(),
)
@handle_cmd_line_args
def test_einsum(*, data, eq_n_op_n_shp, dtype, with_out, tensor_fn, device, call):
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
        if ivy.current_backend_str() in ["tensorflow", "jax"]:
            # these backends do not support native inplace updates
            return
        assert ret.data is out.data
