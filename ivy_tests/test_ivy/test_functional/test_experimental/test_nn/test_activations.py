# global
from hypothesis import assume, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test


# logit
@handle_test(
    fn_tree="functional.ivy.experimental.logit",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
)
def test_logit(
    *,
    dtype_and_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        fw=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
    )


# thresholded_relu
@handle_test(
    fn_tree="functional.ivy.experimental.thresholded_relu",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
    threshold=st.floats(min_value=-0.10, max_value=10.0),
)
def test_thresholded_relu(
    *,
    dtype_and_x,
    threshold,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        fw=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
        threshold=threshold,
    )


@handle_test(
    fn_tree="functional.ivy.experimental.prelu",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=st.shared(helpers.get_shape(), key="prelu"),
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
    slope=helpers.array_values(
        dtype=helpers.get_dtypes("float"),
        shape=st.shared(helpers.get_shape(), key="prelu"),
    ),
)
def test_prelu(
    *,
    dtype_and_x,
    slope,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        fw=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
        slope=slope,
    )


# relu6
@handle_test(
    fn_tree="functional.ivy.experimental.relu6",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        large_abs_safety_factor=2,
        small_abs_safety_factor=2,
        safety_factor_scale="log",
    ),
)
def test_relu6(
    *,
    dtype_and_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        fw=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
    )


@st.composite
def _batch_norm_helper(draw):
    x_dtype, x, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
            min_num_dims=3,
            max_num_dims=5,
            min_dim_size=5,
            ret_shape=True,
            max_value=1000,
            min_value=-1000,
        )
    )
    _, variance = draw(
        helpers.dtype_and_values(
            dtype=x_dtype,
            shape=(shape[1],),
            max_value=1000,
            min_value=0,
        )
    )
    _, others = draw(
        helpers.dtype_and_values(
            dtype=x_dtype * 3,
            shape=(shape[1],),
            max_value=1000,
            min_value=-1000,
            num_arrays=3,
        )
    )
    return x_dtype, x[0], others[0], others[1], others[2], variance[0]


# batch_norm
@handle_test(
    fn_tree="functional.ivy.experimental.batch_norm",
    data=_batch_norm_helper(),
    eps=helpers.floats(min_value=1e-5, max_value=0.1),
    test_with_out=st.just(False),
)
def test_batch_norm(
    *,
    data,
    eps,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    x_dtype, x, scale, offset, mean, variance = data
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        fw=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        xs_grad_idxs=[[0, 0]],
        input_dtypes=x_dtype,
        x=x,
        mean=mean,
        variance=variance,
        scale=scale,
        offset=offset,
        eps=eps,
        rtol_=1e-03,
    )


# sigmoid
@handle_test(
    fn_tree="functional.ivy.experimental.sigmoid",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
)
def test_sigmoid(
    *,
    dtype_and_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        fw=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-2,
        atol_=1e-2,
        x=x[0],
    )


# hard_sigmoid
@handle_test(
    fn_tree="functional.ivy.experimental.hard_sigmoid",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
)
def test_hard_sigmoid(
    *,
    dtype_and_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        fw=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-2,
        atol_=1e-2,
        x=x[0],
    )


# hard_tanh
@handle_test(
    fn_tree="functional.ivy.experimental.hard_tanh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
)
def test_hard_tanh(
    *,
    dtype_and_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        fw=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-2,
        atol_=1e-2,
        x=x[0],
    )


# log_sigmoid
@handle_test(
    fn_tree="functional.ivy.experimental.log_sigmoid",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
)
def test_log_sigmoid(
    *,
    dtype_and_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        fw=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-2,
        atol_=1e-2,
        x=x[0],
    )


# selu
@handle_test(
    fn_tree="functional.ivy.experimental.selu",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
)
def test_selu(
    *,
    dtype_and_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        fw=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-2,
        atol_=1e-2,
        x=x[0],
    )


# softplus
@handle_test(
    fn_tree="functional.ivy.experimental.softplus",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
    beta=st.one_of(helpers.number(min_value=0.1, max_value=10), st.none()),
    threshold=st.one_of(helpers.number(min_value=0.1, max_value=30), st.none()),
)
def test_softplus(
    *,
    dtype_and_x,
    beta,
    threshold,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    assume(beta != 0)
    assume(threshold != 0)
    dtype, x = dtype_and_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        fw=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-02,
        atol_=1e-02,
        x=x[0],
        beta=beta,
        threshold=threshold,
    )


# softsign
@handle_test(
    fn_tree="functional.ivy.experimental.softsign",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
)
def test_softsign(
    *,
    dtype_and_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        fw=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-02,
        atol_=1e-02,
        x=x[0],
    )


# silu
@handle_test(
    fn_tree="functional.ivy.experimental.silu",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
)
def test_silu(
    *,
    dtype_and_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        fw=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-02,
        atol_=1e-02,
        x=x[0],
    )


# hard_silu
@handle_test(
    fn_tree="functional.ivy.experimental.hard_silu",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
)
def test_hard_silu(
    *,
    dtype_and_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        fw=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-02,
        atol_=1e-02,
        x=x[0],
    )


# leaky_relu
@handle_test(
    fn_tree="functional.ivy.experimental.leaky_relu",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", full=False, key="leaky_relu"),
        large_abs_safety_factor=16,
        small_abs_safety_factor=16,
        safety_factor_scale="log",
    ),
    alpha=st.floats(min_value=1e-4, max_value=1e-2),
)
def test_leaky_relu(
    *,
    dtype_and_x,
    alpha,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        fw=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-1,
        atol_=1e-1,
        x=x[0],
        alpha=alpha,
    )


# elu
@handle_test(
    fn_tree="functional.ivy.experimental.elu",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", full=False, key="leaky_relu"),
        large_abs_safety_factor=16,
        small_abs_safety_factor=16,
        safety_factor_scale="log",
    ),
    alpha=st.floats(min_value=1e-4, max_value=1.0),
)
def test_elu(
    *,
    dtype_and_x,
    alpha,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        fw=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-1,
        atol_=1e-1,
        x=x[0],
        alpha=alpha,
    )


# celu
@handle_test(
    fn_tree="functional.ivy.experimental.celu",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", full=False, key="leaky_relu"),
        large_abs_safety_factor=16,
        small_abs_safety_factor=16,
        safety_factor_scale="log",
    ),
    alpha=st.floats(min_value=1e-4, max_value=1.0),
)
def test_celu(
    *,
    dtype_and_x,
    alpha,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        fw=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-1,
        atol_=1e-1,
        x=x[0],
        alpha=alpha,
    )


# glu
@handle_test(
    fn_tree="functional.ivy.experimental.glu",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        valid_axis=True,
        max_axes_size=1,
        force_int_axis=True,
    ).filter(lambda x: x[2] is not None and x[1][0].shape[x[2]] % 2 == 0),
)
def test_glu(
    *,
    dtype_x_axis,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtypes, x, axis = dtype_x_axis
    axis = axis if isinstance(axis, int) else -1
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtypes,
        fw=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
        axis=axis,
        rtol_=1e-1,
        atol_=1e-1,
    )
