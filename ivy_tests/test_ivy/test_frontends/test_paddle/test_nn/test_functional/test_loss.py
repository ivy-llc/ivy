# global
from hypothesis import strategies as st
import ivy

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# --- Helpers --- #
# --------------- #


# cosine embedding loss
@st.composite
def _cos_embd_loss_helper(draw):
    dtype_inputs_shape = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            min_num_dims=1,
            max_num_dims=2,
            min_dim_size=2,
            ret_shape=True,
            num_arrays=2,
        )
    )

    input_dtypes, inputs, shape = dtype_inputs_shape

    _, label = draw(
        helpers.dtype_and_values(
            dtype=input_dtypes, shape=(shape[0],), min_value=-1, max_value=1
        ),
    )

    return input_dtypes, inputs, label


# --- Main --- #
# ------------ #


@handle_frontend_test(
    fn_tree="paddle.nn.functional.binary_cross_entropy",
    dtype_and_vals=helpers.dtype_and_values(
        available_dtypes=["float32", "float64"],
        num_arrays=3,
        min_value=1.0013580322265625e-05,
        max_value=1,
        large_abs_safety_factor=2,
        small_abs_safety_factor=2,
        safety_factor_scale="linear",
        allow_inf=False,
        exclude_min=True,
        exclude_max=True,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
    ),
    reduction=st.sampled_from(["mean", "sum", "none"]),
)
def test_paddle_binary_cross_entropy(
    dtype_and_vals,
    reduction,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x = dtype_and_vals
    helpers.test_frontend_function(
        input_dtypes=[input_dtype[0], input_dtype[1]],
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        label=x[1],
        weight=x[2],
        reduction=reduction,
        rtol=1e-02,
        atol=1e-02,
    )


@handle_frontend_test(
    fn_tree="paddle.nn.functional.binary_cross_entropy_with_logits",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        exclude_min=True,
        exclude_max=True,
        shared_dtype=True,
        min_num_dims=1,
    ),
    dtype_and_weight=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
    ),
    reduction=st.sampled_from(["mean", "none", "sum"]),
)
def test_paddle_binary_cross_entropy_with_logits(
    dtype_and_x,
    dtype_and_weight,
    reduction,
    on_device,
    fn_tree,
    backend_fw,
    frontend,
    test_flags,
):
    # TODO: paddle's implementation of pos_weight is wrong
    # https://github.com/PaddlePaddle/Paddle/pull/54869
    x_dtype, x = dtype_and_x
    weight_dtype, weight = dtype_and_weight
    helpers.test_frontend_function(
        input_dtypes=[
            x_dtype[0],
            x_dtype[1],
            weight_dtype[0],
        ],
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        logit=x[0],
        label=x[1],
        weight=weight[0],
        reduction=reduction,
    )


@handle_frontend_test(
    fn_tree="paddle.nn.functional.cosine_embedding_loss",
    dtype_xs_label=_cos_embd_loss_helper(),
    margin=st.floats(
        min_value=-1.0,
        max_value=1.0,
        width=16,
    ),
    reduction=st.sampled_from(["none", "mean", "sum"]),
    test_with_out=st.just(False),
)
def test_paddle_cosine_embedding_loss(
    *,
    dtype_xs_label,
    margin,
    reduction,
    test_flags,
    fn_tree,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, xs, label = dtype_xs_label
    input1_dtype, input1 = input_dtypes[0], xs[0]
    input2_dtype, input2 = input_dtypes[1], xs[1]

    helpers.test_frontend_function(
        input_dtypes=[input1_dtype, input2_dtype],
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input1=input1,
        input2=input2,
        label=label[0],
        margin=margin,
        reduction=reduction,
    )


@handle_frontend_test(
    fn_tree="paddle.nn.functional.dice_loss",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=1,
        shared_dtype=False,
        min_num_dims=3,
        min_dim_size=3,
        max_num_dims=3,
        max_dim_size=3,
    ),
    labels=st.lists(
        (
            st.lists(
                (
                    st.lists(
                        st.integers(min_value=0, max_value=1), min_size=3, max_size=3
                    )
                ),
                min_size=3,
                max_size=3,
            )
        ),
        min_size=1,
        max_size=1,
    ),
    epsilon=st.floats(
        min_value=1e-6,
        max_value=1e-2,
    ),
)
def test_paddle_dice_loss(
    dtype_and_x,
    labels,
    epsilon,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    x_dtype, x = dtype_and_x
    x[0] = x[0].reshape([3, 3, 3])
    labels = ivy.array(labels, dtype=ivy.int64)
    labels = labels.reshape([3, 3, 1])
    helpers.test_frontend_function(
        input_dtypes=[ivy.int64] + [ivy.float64] + x_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        label=labels,
        epsilon=epsilon,
    )


@handle_frontend_test(
    fn_tree="paddle.nn.functional.hinge_embedding_loss",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        shared_dtype=True,
    ),
    margin=st.floats(
        min_value=-1.0,
        max_value=1.0,
        width=16,
    ),
    reduction=st.sampled_from(["none", "mean", "sum"]),
)
def test_paddle_hinge_embedding_loss(
    dtype_and_x,
    margin,
    reduction,
    test_flags,
    backend_fw,
    fn_tree,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        label=x[1],
        margin=margin,
        reduction=reduction,
    )


@handle_frontend_test(
    fn_tree="paddle.nn.functional.kl_div",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
        min_value=1.0013580322265625e-05,
    ),
    reduction=st.sampled_from(["mean", "batchmean", "sum", "none"]),
)
def test_paddle_kl_div(
    dtype_and_x, reduction, on_device, backend_fw, fn_tree, frontend, test_flags
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        label=x[1],
        reduction=reduction,
    )


@handle_frontend_test(
    fn_tree="paddle.nn.functional.l1_loss",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    reduction=st.sampled_from(["mean", "sum", "none"]),
)
def test_paddle_l1_loss(
    dtype_and_x,
    reduction,
    on_device,
    fn_tree,
    frontend,
    test_flags,
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
        input=x[0],
        label=x[1],
        reduction=reduction,
    )


# log_loss
@handle_frontend_test(
    fn_tree="paddle.nn.functional.log_loss",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        min_value=0,
        max_value=1,
        exclude_min=True,
        exclude_max=True,
        shared_dtype=True,
        min_num_dims=2,
        max_num_dims=2,
        max_dim_size=1,
    ),
    epsilon=st.floats(
        min_value=1e-7,
        max_value=1.0,
    ),
)
def test_paddle_log_loss(
    dtype_and_x,
    epsilon,
    fn_tree,
    test_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        label=x[1],
        epsilon=epsilon,
    )


@handle_frontend_test(
    fn_tree="paddle.nn.functional.margin_ranking_loss",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=3,
        shared_dtype=True,
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    margin=st.floats(
        min_value=-1.0,
        max_value=1.0,
        width=16,
    ),
    reduction=st.sampled_from(["mean", "sum", "none"]),
)
def test_paddle_margin_ranking_loss(
    dtype_and_x,
    margin,
    reduction,
    on_device,
    fn_tree,
    frontend,
    test_flags,
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
        input=x[0],
        other=x[1],
        label=x[2],
        margin=margin,
        reduction=reduction,
    )


# mse_loss
@handle_frontend_test(
    fn_tree="paddle.nn.functional.mse_loss",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
        min_num_dims=1,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
    ),
    reduction=st.sampled_from(["mean", "none", "sum"]),
)
def test_paddle_mse_loss(
    dtype_and_x,
    reduction,
    on_device,
    backend_fw,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        label=x[1],
        reduction=reduction,
    )


@handle_frontend_test(
    fn_tree="paddle.nn.functional.multi_label_soft_margin_loss",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=-2,
        max_value=2,
        shared_dtype=True,
        allow_inf=False,
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtype_and_weight=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=2,
        min_value=-2,
        max_value=2,
    ),
    reduction=st.sampled_from(["mean", "none", "sum"]),
)
def test_paddle_multi_label_soft_margin_loss(
    dtype_and_x,
    dtype_and_weight,
    reduction,
    on_device,
    fn_tree,
    backend_fw,
    frontend,
    test_flags,
):
    x_dtype, x = dtype_and_x
    weight_dtype, weight = dtype_and_weight
    helpers.test_frontend_function(
        input_dtypes=[
            x_dtype[0],
            x_dtype[1],
            weight_dtype[0],
        ],
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        label=x[1],
        weight=weight[0],
        reduction=reduction,
    )


@handle_frontend_test(
    fn_tree="paddle.nn.functional.nll_loss",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
        min_num_dims=1,
        max_num_dims=2,
    ),
    dtype_and_weight=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=1,
        max_num_dims=1,
    ),
    ignore_index=st.integers(
        min_value=-100,
    ),
    reduction=st.sampled_from(["mean", "sum", "none"]),
)
def test_paddle_nll_loss(
    dtype_and_x,
    dtype_and_weight,
    ignore_index,
    reduction,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    x_dtype, x = dtype_and_x
    weight_dtype, weight = dtype_and_weight
    helpers.test_frontend_function(
        input_dtypes=[
            x_dtype[0],
            x_dtype[1],
            weight_dtype[0],
        ],
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        label=x[1],
        weight=weight[0],
        ignore_index=ignore_index,
        reduction=reduction,
    )


@handle_frontend_test(
    fn_tree="paddle.nn.functional.sigmoid_focal_loss",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=1,
        shared_dtype=False,
        min_num_dims=1,
        min_dim_size=1,
    ),
    dtype_and_normalizer=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=1,
        shared_dtype=True,
        min_num_dims=1,
        min_dim_size=1,
        max_num_dims=1,
        max_dim_size=1,
    ),
    dtype_and_labels=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=1,
        shared_dtype=False,
        min_num_dims=1,
        min_dim_size=1,
        min_value=0,
        max_value=1,
    ),
    alpha=st.floats(
        min_value=0.0,
        max_value=1.0,
    ),
    gamma=st.floats(
        min_value=0.0,
        max_value=5.0,
    ),
    reduction=st.sampled_from(["mean", "sum", "none"]),
)
def test_paddle_sigmoid_focal_loss(
    dtype_and_x,
    dtype_and_normalizer,
    dtype_and_labels,
    alpha,
    gamma,
    reduction,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    x_dtype, x = dtype_and_x
    normalizer_dtype, normalizer = dtype_and_normalizer
    label_dtype, labels = dtype_and_labels
    normalizer = [norm.reshape(-1) for norm in normalizer]
    labels = ivy.array(labels, dtype=ivy.int64)
    helpers.test_frontend_function(
        input_dtypes=[ivy.int64]
        + [ivy.float64]
        + x_dtype
        + normalizer_dtype
        + label_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        logit=x[0],
        label=labels[0],
        alpha=alpha,
        gamma=gamma,
        normalizer=normalizer[0],
        reduction=reduction,
    )


# smooth_l1_loss
@handle_frontend_test(
    fn_tree="paddle.nn.functional.smooth_l1_loss",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        shared_dtype=True,
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    delta=st.floats(
        min_value=0.1,
        max_value=1.0,
    ),
    reduction=st.sampled_from(["mean", "sum", "none"]),
)
def test_paddle_smooth_l1_loss(
    dtype_and_x,
    delta,
    reduction,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        label=x[1],
        reduction=reduction,
        delta=delta,
    )


@handle_frontend_test(
    fn_tree="paddle.nn.functional.softmax_with_cross_entropy",
    dtype_and_x_and_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=1e-04,
        max_value=1,
        min_num_dims=2,
        allow_inf=False,
        shared_dtype=True,
        force_int_axis=True,
        valid_axis=True,
    ),
    soft_label=st.booleans(),
    numeric_stable_mode=st.booleans(),
    return_softmax=st.booleans(),
)
def test_paddle_softmax_with_cross_entropy(
    dtype_and_x_and_axis,
    soft_label,
    numeric_stable_mode,
    return_softmax,
    on_device,
    fn_tree,
    backend_fw,
    frontend,
    test_flags,
):
    x_dtype, x, axis = dtype_and_x_and_axis
    logits = x[0]
    labels = x[1]
    label_dtype = x_dtype
    ignore_index = 0
    if soft_label:
        labels = labels / ivy.sum(labels).to_native()
    else:
        labels = ivy.argmax(labels, axis=axis).to_native()
        flattened_labels = labels.flatten()
        ignore_index = ivy.randint(0, flattened_labels.size)
        ignore_index = flattened_labels[ignore_index]
        label_dtype = [str(labels.dtype)]
    if on_device == "cpu" or soft_label:
        numeric_stable_mode = True
    helpers.test_frontend_function(
        input_dtypes=[x_dtype[0], label_dtype[0]],
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        logits=logits,
        label=labels,
        soft_label=soft_label,
        ignore_index=ignore_index,
        numeric_stable_mode=numeric_stable_mode,
        return_softmax=return_softmax,
        axis=axis,
    )


@handle_frontend_test(
    fn_tree="paddle.nn.functional.loss.square_error_cost",
    dtype_and_input_and_label=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"), num_arrays=2
    ),
)
def test_paddle_square_error_cost(
    *,
    dtype_and_input_and_label,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
):
    input_dtypes, input_and_label = dtype_and_input_and_label
    input, label = input_and_label
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        input=input,
        label=label,
        fn_tree=fn_tree,
    )


@handle_frontend_test(
    fn_tree="paddle.nn.functional.triplet_margin_loss",
    dtype_and_inputs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=3,
        allow_inf=False,
        shared_dtype=True,
        min_value=0.0,
        max_value=1.0,
        min_num_dims=1,
        max_num_dims=2,
        min_dim_size=1,
    ),
    margin=st.floats(min_value=1e-6, max_value=1e6),
    p=st.integers(min_value=0, max_value=2),
    swap=st.booleans(),
    reduction=st.sampled_from(["none", "mean", "sum"]),
    test_with_out=st.just(False),
)
def test_paddle_triplet_margin_loss(
    dtype_and_inputs,
    margin,
    p,
    swap,
    reduction,
    test_flags,
    fn_tree,
    backend_fw,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_inputs
    helpers.test_frontend_function(
        input_dtypes=[input_dtype[0], input_dtype[1], input_dtype[2]],
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        positive=x[1],
        negative=x[2],
        margin=margin,
        p=p,
        swap=swap,
        reduction=reduction,
    )
