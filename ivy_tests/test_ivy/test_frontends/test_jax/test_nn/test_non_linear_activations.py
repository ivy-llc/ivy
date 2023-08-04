from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# Todo : turn on complex dtype activation tests once supported in all backends
@handle_frontend_test(
    fn_tree="jax.nn.relu",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_integer"),
        large_abs_safety_factor=3,
        small_abs_safety_factor=3,
        safety_factor_scale="linear",
    ),
    test_with_out=st.just(False),
)
def test_jax_relu(
    *,
    dtype_and_x,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


@handle_frontend_test(
    fn_tree="jax.nn.relu6",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_integer"),
        large_abs_safety_factor=2,
        small_abs_safety_factor=2,
        safety_factor_scale="linear",
    ),
    test_with_out=st.just(False),
)
def test_jax_relu6(
    *,
    dtype_and_x,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


@handle_frontend_test(
    fn_tree="jax.nn.soft_sign",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_integer"),
        large_abs_safety_factor=2,
        small_abs_safety_factor=2,
        safety_factor_scale="linear",
    ),
    test_with_out=st.just(False),
)
def test_jax_soft_sign(
    *,
    dtype_and_x,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


@handle_frontend_test(
    fn_tree="jax.nn.silu",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_integer"),
        large_abs_safety_factor=2,
        small_abs_safety_factor=2,
        safety_factor_scale="linear",
    ),
    test_with_out=st.just(False),
)
def test_jax_silu(
    *,
    dtype_and_x,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


@handle_frontend_test(
    fn_tree="jax.nn.leaky_relu",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        large_abs_safety_factor=2,
        small_abs_safety_factor=2,
        safety_factor_scale="linear",
    ),
    negative_slope=helpers.floats(min_value=0.0, max_value=1.0),
    test_with_out=st.just(False),
)
def test_jax_leaky_relu(
    *,
    dtype_and_x,
    negative_slope,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        rtol=1e-01,
        atol=1e-01,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        negative_slope=negative_slope,
    )


@handle_frontend_test(
    fn_tree="jax.nn.gelu",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
    approximate=st.booleans(),
    test_with_out=st.just(False),
)
def test_jax_gelu(
    *,
    dtype_and_x,
    approximate,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-02,
        atol=1e-02,
        x=x[0],
        approximate=approximate,
    )


@handle_frontend_test(
    fn_tree="jax.nn.sigmoid",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=2,
        small_abs_safety_factor=2,
        safety_factor_scale="linear",
    ),
    test_with_out=st.just(False),
)
def test_jax_sigmoid(
    *,
    dtype_and_x,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# one_hot
@st.composite
def _dtype_indices_classes_axis(draw):
    classes = draw(helpers.ints(min_value=2, max_value=100))
    dtype, indices, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("integer"),
            min_value=0,
            max_value=classes - 1,
            small_abs_safety_factor=4,
            ret_shape=True,
        )
    )

    axis = draw(st.integers(min_value=-1, max_value=len(shape) - 1))
    return dtype, indices, classes, axis


@handle_frontend_test(
    fn_tree="jax.nn.one_hot",
    dtype_indices_classes_axis=_dtype_indices_classes_axis(),
    dtype=helpers.get_dtypes("float", full=False),
    test_with_out=st.just(False),
)
def test_jax_one_hot(
    *,
    dtype_indices_classes_axis,
    dtype,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    input_dtype, indices, num_classes, axis = dtype_indices_classes_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-02,
        x=indices[0],
        num_classes=num_classes,
        dtype=dtype[0],
        axis=axis,
    )


@handle_frontend_test(
    fn_tree="jax.nn.softmax",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=2,
        max_axes_size=1,
        force_int_axis=True,
        valid_axis=True,
    ),
    test_with_out=st.just(False),
)
def test_jax_softmax(
    *,
    dtype_x_axis,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    x_dtype, x, axis = dtype_x_axis

    helpers.test_frontend_function(
        input_dtypes=x_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-02,
        atol=1e-02,
        x=x[0],
        axis=axis,
    )


@handle_frontend_test(
    fn_tree="jax.nn.softplus",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_integer"),
        large_abs_safety_factor=2,
        small_abs_safety_factor=2,
        safety_factor_scale="linear",
    ),
    test_with_out=st.just(False),
)
def test_jax_softplus(
    *,
    dtype_and_x,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


@handle_frontend_test(
    fn_tree="jax.nn.log_sigmoid",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-100,
        max_value=100,
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
    test_with_out=st.just(False),
)
def test_jax_log_sigmoid(
    *,
    dtype_and_x,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        rtol=1e-02,
        atol=1e-02,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


@handle_frontend_test(
    fn_tree="jax.nn.log_softmax",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=2,
        small_abs_safety_factor=2,
        safety_factor_scale="linear",
        min_value=-2,
        min_num_dims=1,
    ),
    axis=helpers.ints(min_value=-1, max_value=0),
    test_with_out=st.just(False),
)
def test_jax_log_softmax(
    *,
    dtype_and_x,
    axis,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-02,
        atol=1e-02,
        x=x[0],
        axis=axis,
    )


# glu
@handle_frontend_test(
    fn_tree="jax.nn.glu",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=2,
        small_abs_safety_factor=2,
        safety_factor_scale="linear",
        min_value=-2,
        min_num_dims=1,
        min_dim_size=4,
        max_dim_size=4,
    ),
    axis=helpers.ints(min_value=-1, max_value=0),
    test_with_out=st.just(False),
)
def test_jax_glu(
    *,
    dtype_and_x,
    axis,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-01,
        atol=1e-01,
        x=x[0],
        axis=axis,
    )


@handle_frontend_test(
    fn_tree="jax.nn.normalize",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_integer"),
        large_abs_safety_factor=2,
        small_abs_safety_factor=2,
        safety_factor_scale="linear",
        num_arrays=3,
        shared_dtype=True,
    ),
    axis=st.just(-1),
    epsilon=helpers.floats(
        min_value=0.01,
        max_value=1,
    ),
    where=st.none(),
    test_with_out=st.just(False),
)
def test_jax_normalize(
    *,
    dtype_and_x,
    axis,
    epsilon,
    where,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    input_dtypes, xs = dtype_and_x

    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        rtol=1e-02,
        atol=1e-02,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xs[0],
        axis=axis,
        mean=xs[1],
        variance=xs[2],
        epsilon=epsilon,
        where=where,
    )


@handle_frontend_test(
    fn_tree="jax.nn.hard_tanh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_integer"),
        large_abs_safety_factor=2,
        small_abs_safety_factor=2,
        safety_factor_scale="linear",
    ),
    test_with_out=st.just(False),
)
def test_jax_hard_tanh(
    *,
    dtype_and_x,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    # TODO: enable this test for all valid dtypes as jax.nn.hard_tanh supports
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


@handle_frontend_test(
    fn_tree="jax.nn.celu",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_integer"),
        min_value=-5,
        max_value=5,
        safety_factor_scale="linear",
    ),
    alpha=helpers.floats(min_value=0.01, max_value=1),
    test_with_out=st.just(False),
)
def test_jax_celu(
    *,
    dtype_and_x,
    alpha,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    input_dtypes, xs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xs[0],
        alpha=alpha,
    )


# elu
@handle_frontend_test(
    fn_tree="jax.nn.elu",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_integer"),
        min_value=-5,
        max_value=5,
        safety_factor_scale="linear",
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_jax_elu(
    *,
    dtype_and_x,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    input_dtypes, xs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xs[0],
        alpha=xs[1],
        rtol=1e-03,
        atol=1e-03,
    )


@handle_frontend_test(
    fn_tree="jax.nn.logsumexp",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_integer"),
        large_abs_safety_factor=2,
        small_abs_safety_factor=2,
        safety_factor_scale="linear",
        num_arrays=2,
        shared_dtype=True,
    ),
    axis=st.just(None),
    keepdims=st.booleans(),
    return_sign=st.booleans(),
    test_with_out=st.just(False),
)
def test_jax_logsumexp(
    *,
    dtype_and_x,
    axis,
    keepdims,
    return_sign,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    input_dtypes, xs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=xs[0],
        axis=axis,
        b=xs[1],
        keepdims=keepdims,
        return_sign=return_sign,
    )


@handle_frontend_test(
    fn_tree="jax.nn.swish",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=2,
        small_abs_safety_factor=2,
        safety_factor_scale="linear",
    ),
    test_with_out=st.just(False),
)
def test_jax_swish(
    *,
    dtype_and_x,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


@handle_frontend_test(
    fn_tree="jax.nn.hard_swish",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-10,
        max_value=10,
        safety_factor_scale="linear",
    ),
    test_with_out=st.just(False),
)
def test_jax_hard_swish(
    *,
    dtype_and_x,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        rtol=1e-02,
        atol=1e-02,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


@handle_frontend_test(
    fn_tree="jax.nn.hard_silu",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_integer"),
        large_abs_safety_factor=2,
        small_abs_safety_factor=2,
    ),
    test_with_out=st.just(False),
)
def test_jax_hard_silu(
    *,
    dtype_and_x,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    input_dtypes, xs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xs[0],
    )


@handle_frontend_test(
    fn_tree="jax.nn.hard_sigmoid",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_integer"),
        large_abs_safety_factor=2,
        small_abs_safety_factor=2,
    ),
    test_with_out=st.just(False),
)
def test_jax_hard_sigmoid(
    *,
    dtype_and_x,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    input_dtypes, xs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=xs[0],
    )


@handle_frontend_test(
    fn_tree="jax.nn.selu",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_integer"),
        large_abs_safety_factor=2,
        small_abs_safety_factor=2,
        safety_factor_scale="log",
    ),
    test_with_out=st.just(False),
)
def test_jax_selu(
    *,
    dtype_and_x,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-02,
        atol=1e-02,
        x=x[0],
    )
