import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        large_value_safety_factor=1,
        small_value_safety_factor=1,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.nn.relu"
    ),
)
def test_jax_nn_relu(
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
        fn_tree="nn.relu",
        x=np.asarray(x, dtype=input_dtype),
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        large_value_safety_factor=1,
        small_value_safety_factor=1,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.nn.relu6"
    ),
)
def test_jax_nn_relu6(
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
        fn_tree="nn.relu6",
        x=np.asarray(x, dtype=input_dtype),
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        large_value_safety_factor=1,
        small_value_safety_factor=1,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.nn.soft_sign"
    ),
)
def test_jax_nn_soft_sign(
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
        fn_tree="nn.soft_sign",
        x=np.asarray(x, dtype=input_dtype),
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        large_value_safety_factor=1,
        small_value_safety_factor=1,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.nn.silu"
    ),
)
def test_jax_nn_silu(
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
        fn_tree="nn.silu",
        x=np.asarray(x, dtype=input_dtype),
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        large_value_safety_factor=1,
        small_value_safety_factor=1,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.nn.leaky_relu"
    ),
    negative_slope=helpers.floats(min_value=0.0, max_value=1.0),
)
def test_jax_nn_leaky_relu(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    fw,
    negative_slope,
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
        fn_tree="nn.leaky_relu",
        x=np.asarray(x, dtype=input_dtype),
        negative_slope=negative_slope,
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_value_safety_factor=1,
        small_value_safety_factor=1,
    ),
    approximate=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.nn.gelu"
    ),
)
def test_jax_nn_gelu(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    fw,
    approximate,
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
        fn_tree="nn.gelu",
        x=np.asarray(x, dtype=input_dtype),
        approximate=approximate,
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        large_value_safety_factor=1,
        small_value_safety_factor=1,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.nn.sigmoid"
    ),
)
def test_jax_nn_sigmoid(
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
        fn_tree="nn.sigmoid",
        x=np.asarray(x, dtype=input_dtype),
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        large_value_safety_factor=1,
        small_value_safety_factor=1,
        min_value=1,
        max_value=3,
    ),
    num_classes=st.integers(min_value=4, max_value=6),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.nn.one_hot"
    ),
)
def test_jax_nn_one_hot(
    dtype_and_x,
    num_classes,
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
        fn_tree="nn.one_hot",
        x=np.asarray(x, dtype=input_dtype),
        num_classes=num_classes,
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_value_safety_factor=1,
        small_value_safety_factor=1,
        min_value=-2,
        min_num_dims=1,
    ),
    axis=helpers.ints(min_value=-1, max_value=0),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.nn.softmax"
    ),
)
def test_jax_nn_softmax(
    dtype_and_x,
    axis,
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
        fn_tree="nn.softmax",
        x=np.asarray(x, dtype=input_dtype),
        axis=axis,
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        large_value_safety_factor=1,
        small_value_safety_factor=1,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.nn.softplus"
    ),
)
def test_jax_nn_softplus(
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
        fn_tree="nn.softplus",
        x=np.asarray(x, dtype=input_dtype),
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_value_safety_factor=1,
        small_value_safety_factor=1,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.nn.log_sigmoid"
    ),
)
def test_jax_nn_log_sigmoid(
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
        fn_tree="nn.log_sigmoid",
        x=np.asarray(x, dtype=input_dtype),
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_value_safety_factor=1,
        small_value_safety_factor=1,
        min_value=-2,
        min_num_dims=1,
    ),
    axis=helpers.ints(min_value=-1, max_value=0),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.nn.log_softmax"
    ),
)
def test_jax_nn_log_softmax(
    dtype_and_x,
    axis,
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
        fn_tree="nn.log_softmax",
        rtol=1e-3,
        atol=1e-3,
        x=np.asarray(x, dtype=input_dtype),
        axis=axis,
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_value_safety_factor=1,
        small_value_safety_factor=1,
        min_value=-2,
        min_num_dims=1,
        min_dim_size=4,
        max_dim_size=4,
    ),
    axis=helpers.ints(min_value=-1, max_value=0),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.nn.glu"
    ),
)
def test_jax_nn_glu(
    dtype_and_x,
    axis,
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
        fn_tree="nn.glu",
        rtol=1e-3,
        atol=1e-3,
        x=np.asarray(x, dtype=input_dtype),
        axis=axis,
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        large_value_safety_factor=1,
        small_value_safety_factor=1,
        num_arrays=3,
        shared_dtype=True,
    ),
    axis=st.just(-1),
    epsilon=helpers.floats(
        min_value=0.01,
        max_value=1,
    ),
    where=st.none(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.nn.normalize"
    ),
)
def test_jax_nn_normalize(
    dtype_and_x,
    axis,
    epsilon,
    where,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtypes, xs = dtype_and_x

    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="nn.normalize",
        x=np.asarray(xs[0], dtype=input_dtypes[0]),
        axis=axis,
        mean=np.asarray(xs[1], dtype=input_dtypes[1]),
        variance=np.asarray(xs[2], dtype=input_dtypes[2]),
        epsilon=epsilon,
        where=where,
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        large_value_safety_factor=1,
        small_value_safety_factor=1,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.nn.hard_tanh"
    ),
)
def test_jax_nn_hard_tanh(
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
        fn_tree="nn.hard_tanh",
        x=np.asarray(x, dtype=input_dtype),
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        large_value_safety_factor=1,
        small_value_safety_factor=1,
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.nn.celu"
    ),
)
def test_jax_nn_celu(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtypes, xs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="nn.celu",
        x=np.asarray(xs[0], dtype=input_dtypes[0]),
        alpha=np.asarray(xs[1], dtype=input_dtypes[1]),
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        large_value_safety_factor=1,
        small_value_safety_factor=1,
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.nn.elu"
    ),
)
def test_jax_nn_elu(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtypes, xs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="nn.elu",
        x=np.asarray(xs[0], dtype=input_dtypes[0]),
        alpha=np.asarray(xs[1], dtype=input_dtypes[1]),
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        large_value_safety_factor=1,
        small_value_safety_factor=1,
        num_arrays=2,
        shared_dtype=True,
    ),
    axis=st.just(None),
    keepdims=st.booleans(),
    return_sign=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.nn.logsumexp"
    ),
)
def test_jax_nn_logsumexp(
    dtype_and_x,
    axis,
    keepdims,
    return_sign,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtypes, xs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="nn.logsumexp",
        a=np.asarray(xs[0], dtype=input_dtypes[0]),
        axis=axis,
        b=np.asarray(xs[1], dtype=input_dtypes[1]),
        keepdims=keepdims,
        return_sign=return_sign,
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_value_safety_factor=1,
        small_value_safety_factor=1,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.nn.swish"
    ),
)
def test_jax_nn_swish(
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
        fn_tree="nn.swish",
        x=np.asarray(x, dtype=input_dtype),
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_value_safety_factor=1,
        small_value_safety_factor=1,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.nn.hard_swish"
    ),
)
def test_jax_nn_hard_swish(
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
        fn_tree="nn.hard_swish",
        x=np.asarray(x, dtype=input_dtype),
    )
