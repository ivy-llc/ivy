# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
import ivy
from ivy.functional.frontends.torch.nn.functional.loss_functions import (
    cosine_similarity,
)


# cross_entropy
@handle_frontend_test(
    fn_tree="torch.nn.functional.cross_entropy",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=1,
    ),
    dtype_and_target=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0.0,
        max_value=1.0,
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
    ),
    dtype_and_weights=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
    ),
    size_average=st.booleans(),
    reduce=st.booleans(),
    reduction=st.sampled_from(["mean", "none", "sum"]),
    label_smoothing=helpers.floats(min_value=0, max_value=0.49),
)
def test_torch_cross_entropy(
    *,
    dtype_and_input,
    dtype_and_target,
    dtype_and_weights,
    size_average,
    reduce,
    reduction,
    label_smoothing,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    inputs_dtype, input = dtype_and_input
    target_dtype, target = dtype_and_target
    weights_dtype, weights = dtype_and_weights
    helpers.test_frontend_function(
        input_dtypes=inputs_dtype + target_dtype + weights_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input[0],
        target=target[0],
        weight=weights[0],
        size_average=size_average,
        reduce=reduce,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )


# binary_cross_entropy
@handle_frontend_test(
    fn_tree="torch.nn.functional.binary_cross_entropy",
    dtype_and_vals=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=3,
        shared_dtype=True,
        min_value=1.0013580322265625e-05,
        max_value=1.0,
        large_abs_safety_factor=2,
        small_abs_safety_factor=2,
        safety_factor_scale="linear",
        allow_inf=False,
        exclude_min=True,
        exclude_max=True,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
        shape=(5,),
    ),
    size_average=st.booleans(),
    reduce=st.booleans(),
    reduction=st.sampled_from(["mean", "none", "sum", None]),
)
def test_torch_binary_cross_entropy(
    *,
    dtype_and_vals,
    size_average,
    reduce,
    reduction,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_vals
    pred_dtype, pred = input_dtype[0], x[0]
    true_dtype, true = input_dtype[1], x[1]
    weight_dtype, weight = input_dtype[2], x[2]

    helpers.test_frontend_function(
        input_dtypes=[pred_dtype, true_dtype, weight_dtype],
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=pred,
        target=true,
        weight=weight,
        size_average=size_average,
        reduce=reduce,
        reduction=reduction,
        rtol=1e-02,
        atol=1e-02,
    )


# binary_cross_entropy_with_logits
@handle_frontend_test(
    fn_tree="torch.nn.functional.binary_cross_entropy_with_logits",
    dtype_and_true=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0.0,
        max_value=1.0,
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
    dtype_and_pred=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=1.0013580322265625e-05,
        max_value=1.0,
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
    dtype_and_weight=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=1.0013580322265625e-05,
        max_value=1.0,
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
    ),
    size_average=st.booleans(),
    reduce=st.booleans(),
    reduction=st.sampled_from(["mean", "none", "sum", None]),
    dtype_and_pos_weight=st.one_of(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
            min_value=0,
            max_value=10,
            allow_inf=False,
            exclude_min=True,
            exclude_max=True,
            min_num_dims=1,
            max_num_dims=1,
            min_dim_size=2,
        ),
        st.just([[None], [None]]),
    ),
)
def test_torch_binary_cross_entropy_with_logits(
    *,
    dtype_and_true,
    dtype_and_pred,
    dtype_and_weight,
    size_average,
    reduce,
    reduction,
    dtype_and_pos_weight,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    pred_dtype, pred = dtype_and_pred
    true_dtype, true = dtype_and_true
    weight_dtype, weight = dtype_and_weight
    pos_weight_dtype, pos_weight = dtype_and_pos_weight
    helpers.test_frontend_function(
        input_dtypes=[
            pred_dtype[0],
            true_dtype[0],
            weight_dtype[0],
            pos_weight_dtype[0],
        ],
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=pred[0],
        target=true[0],
        weight=weight[0],
        size_average=size_average,
        reduce=reduce,
        reduction=reduction,
        pos_weight=pos_weight[0],
    )


# cosine_embedding_loss
@handle_frontend_test(
    fn_tree="torch.nn.functional.cosine_embedding_loss",
    dtype_and_inputs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=2,
        max_value=5,
        min_num_dims=1,
        max_num_dims=2,
        min_dim_size=2,
        shared_dtype=True,
        num_arrays=2,
    ),
    margin=st.floats(
        min_value=-1.0,
        max_value=1.0,
        width=16,
    ),
    size_average=st.booleans(),
    reduce=st.booleans(),
    reduction=st.sampled_from(["none", "mean", "sum"]),
    test_with_out=st.just(False),
)
def test_torch_cosine_embedding_loss(
    *,
    dtype_and_inputs,
    margin,
    size_average,
    reduce,
    reduction,
    test_flags,
    fn_tree,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_inputs
    input1_dtype, input1 = input_dtype[0], x[0]
    input2_dtype, input2 = input_dtype[1], x[1]

    if input1.ndim == input2.ndim == 1:
        tar = ivy.array(1.0)
    else:
        third = input1.shape[0] // 3
        ones = ivy.ones(input1.shape[0] - (third * 2))
        minus_ones = ivy.ones(third) * -1
        randoms = ivy.random_uniform(shape=[third])
        tar = ivy.hstack((ones, minus_ones, randoms)).shuffle()

    helpers.test_frontend_function(
        input_dtypes=[input1_dtype, input2_dtype],
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input1=input1,
        input2=input2,
        target=tar,
        margin=margin,
        size_average=size_average,
        reduce=reduce,
        reduction=reduction,
    )


# mse_loss
@handle_frontend_test(
    fn_tree="torch.nn.functional.mse_loss",
    dtype_and_true=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0.0,
        max_value=1.0,
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
    dtype_and_pred=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0.0,
        max_value=1.0,
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
    size_average=st.booleans(),
    reduce=st.booleans(),
    reduction=st.sampled_from(["mean"]),
    test_with_out=st.just(False),
)
def test_torch_mse_loss(
    *,
    dtype_and_true,
    dtype_and_pred,
    size_average,
    reduce,
    reduction,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    pred_dtype, pred = dtype_and_pred
    true_dtype, true = dtype_and_true
    helpers.test_frontend_function(
        input_dtypes=[pred_dtype[0], true_dtype[0]],
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        input=pred[0],
        target=true[0],
        size_average=size_average,
        reduce=reduce,
        reduction=reduction,
        on_device=on_device,
    )


# smooth_l1_loss
@handle_frontend_test(
    fn_tree="torch.nn.functional.smooth_l1_loss",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        allow_inf=False,
        shared_dtype=True,
    ),
    size_average=st.booleans(),
    reduce=st.booleans(),
    reduction=st.sampled_from(["none", "mean", "sum"]),
    beta=st.sampled_from([1.0, 0.5, 0.1, 0.0]),
    test_with_out=st.just(False),
)
def test_torch_smooth_l1_loss(
    *,
    dtype_and_x,
    size_average,
    reduce,
    reduction,
    beta,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    pred_dtype, pred = input_dtype[0], x[0]
    true_dtype, true = input_dtype[1], x[1]
    helpers.test_frontend_function(
        input_dtypes=[pred_dtype, true_dtype],
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=pred,
        target=true,
        size_average=size_average,
        reduce=reduce,
        reduction=reduction,
        beta=beta,
    )


# huber_loss
@handle_frontend_test(
    fn_tree="torch.nn.functional.huber_loss",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        allow_inf=False,
        shared_dtype=True,
    ),
    delta=helpers.floats(min_value=0, max_value=5),
    reduction=st.sampled_from(["none", "mean", "sum"]),
    test_with_out=st.just(False),
)
def test_torch_huber_loss(
    *,
    dtype_and_x,
    delta,
    reduction,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    pred_dtype, pred = input_dtype[0], x[0]
    true_dtype, true = input_dtype[1], x[1]
    helpers.test_frontend_function(
        input_dtypes=[pred_dtype, true_dtype],
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=pred,
        target=true,
        reduction=reduction,
        delta=delta,
    )


# l1_loss
@handle_frontend_test(
    fn_tree="torch.nn.functional.l1_loss",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        allow_inf=False,
        shared_dtype=True,
    ),
    size_average=st.booleans(),
    reduce=st.booleans(),
    reduction=st.sampled_from(["none", "mean", "sum"]),
    test_with_out=st.just(False),
)
def test_torch_l1_loss(
    *,
    dtype_and_x,
    size_average,
    reduce,
    reduction,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    pred_dtype, pred = input_dtype[0], x[0]
    true_dtype, true = input_dtype[1], x[1]
    helpers.test_frontend_function(
        input_dtypes=[pred_dtype, true_dtype],
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=pred,
        target=true,
        size_average=size_average,
        reduce=reduce,
        reduction=reduction,
    )


# nll_loss
@handle_frontend_test(
    fn_tree="torch.nn.functional.nll_loss",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0.01,
        max_value=1.0,
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=1,
        max_dim_size=1,
    ),
    dtype_and_target=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        min_value=0.0,
        max_value=1.0,
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=1,
        max_dim_size=1,
    ),
    dtype_and_weights=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=1,
        max_dim_size=1,
    ),
    size_average=st.booleans(),
    reduce=st.booleans(),
    reduction=st.sampled_from(["mean", "none", "sum"]),
)
def test_torch_nll_loss(
    *,
    dtype_and_input,
    dtype_and_target,
    dtype_and_weights,
    size_average,
    reduce,
    reduction,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    inputs_dtype, input = dtype_and_input
    target_dtype, target = dtype_and_target
    weights_dtype, weights = dtype_and_weights
    helpers.test_frontend_function(
        input_dtypes=inputs_dtype + target_dtype + weights_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input[0],
        target=target[0],
        weight=weights[0],
        size_average=size_average,
        reduce=reduce,
        reduction=reduction,
    )


# gaussian_nll_loss
@handle_frontend_test(
    fn_tree="torch.nn.functional.gaussian_nll_loss",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=3,
        min_value=0.01,
        max_value=1.0,
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    full=st.booleans(),
    eps=st.floats(
        min_value=0.0,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    reduction=st.sampled_from(["mean", "sum"]),
)
def test_torch_gaussian_nll_loss(
    *,
    dtype_and_input,
    full,
    eps,
    reduction,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    inputs_dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=inputs_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input[0],
        target=input[1],
        var=input[2],
        full=full,
        eps=eps,
        reduction=reduction,
        atol=1e-2,
        rtol=1e-2,
    )


# soft margin loss
@handle_frontend_test(
    fn_tree="torch.nn.functional.soft_margin_loss",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        allow_inf=False,
        shared_dtype=True,
    ),
    size_average=st.booleans(),
    reduce=st.booleans(),
    reduction=st.sampled_from(["none", "mean", "sum"]),
    test_with_out=st.just(False),
)
def test_torch_soft_margin_loss(
    *,
    dtype_and_x,
    size_average,
    reduce,
    reduction,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    pred_dtype, pred = input_dtype[0], x[0]
    tar_dtype, tar = input_dtype[1], x[1]
    helpers.test_frontend_function(
        input_dtypes=[pred_dtype, tar_dtype],
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=pred,
        target=tar,
        size_average=size_average,
        reduce=reduce,
        reduction=reduction,
    )


# kl_div
@handle_frontend_test(
    fn_tree="torch.nn.functional.kl_div",
    dtype_and_inputs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
        shared_dtype=True,
        min_value=0,
        max_value=10,
        min_num_dims=0,
        max_num_dims=10,
        min_dim_size=0,
        max_dim_size=10,
        num_arrays=2,
    ),
    size_average=st.booleans(),
    reduce=st.booleans(),
    reduction=st.sampled_from(["none", "mean", "sum", "batchmean"]),
    log_target=st.booleans(),
    test_with_out=st.just(False),
)
def test_torch_kl_div(
    *,
    dtype_and_inputs,
    size_average,
    reduce,
    reduction,
    log_target,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    inputs_dtype, inputs = dtype_and_inputs
    helpers.test_frontend_function(
        input_dtypes=inputs_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=inputs[0],
        target=inputs[1],
        size_average=size_average,
        reduce=reduce,
        reduction=reduction,
        log_target=log_target,
    )


# margin ranking loss
@handle_frontend_test(
    fn_tree="torch.nn.functional.margin_ranking_loss",
    dtype_and_inputs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=3,
        allow_inf=False,
        shared_dtype=True,
    ),
    margin=st.floats(),
    size_average=st.booleans(),
    reduce=st.booleans(),
    reduction=st.sampled_from(["none", "mean", "sum"]),
    test_with_out=st.just(False),
)
def test_torch_margin_ranking_loss(
    *,
    dtype_and_inputs,
    margin,
    size_average,
    reduce,
    reduction,
    test_flags,
    fn_tree,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_inputs
    input1_dtype, input1 = input_dtype[0], x[0]
    input2_dtype, input2 = input_dtype[1], x[1]
    tar_dtype, tar = input_dtype[2], x[2]
    helpers.test_frontend_function(
        input_dtypes=[input1_dtype, input2_dtype, tar_dtype],
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input1=input1,
        input2=input2,
        target=tar,
        margin=margin,
        size_average=size_average,
        reduce=reduce,
        reduction=reduction,
    )


# poisson_nll_loss
@handle_frontend_test(
    fn_tree="torch.nn.functional.poisson_nll_loss",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=0.0,
        max_value=1.0,
        allow_inf=False,
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=1,
    ),
    log_input=st.booleans(),
    full=st.booleans(),
    size_average=st.booleans(),
    reduce=st.booleans(),
    reduction=st.sampled_from(["mean", "none", "sum"]),
)
def test_torch_poisson_nll_loss(
    *,
    dtype_and_input,
    log_input,
    full,
    size_average,
    reduce,
    reduction,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    inputs_dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=inputs_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input[0],
        target=input[1],
        log_input=log_input,
        full=full,
        size_average=size_average,
        reduce=reduce,
        reduction=reduction,
    )


@handle_frontend_test(
    fn_tree="torch.nn.functional.hinge_embedding_loss",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=-100,
        max_value=100,
        allow_inf=False,
    ),
    margin=st.floats(min_value=-10, max_value=10),
    size_average=st.booleans(),
    reduce=st.booleans(),
    reduction=st.sampled_from(["none", "mean", "sum"]),
    test_with_out=st.just(False),
)
def test_torch_hinge_embedding_loss(
    *,
    dtype_and_x,
    margin,
    size_average,
    reduce,
    reduction,
    test_flags,
    fn_tree,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_x
    input, target = x

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input,
        target=target,
        margin=margin,
        size_average=size_average,
        reduce=reduce,
        reduction=reduction,
        atol=1e-5,
        rtol=1e-5,
    )


# triplet margin loss
@handle_frontend_test(
    fn_tree="torch.nn.functional.triplet_margin_loss",
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
    margin=st.floats(),
    p=st.integers(min_value=0, max_value=2),
    swap=st.booleans(),
    size_average=st.booleans(),
    reduce=st.booleans(),
    reduction=st.sampled_from(["none", "mean", "sum"]),
    test_with_out=st.just(False),
)
def test_torch_triplet_margin_loss(
    *,
    dtype_and_inputs,
    margin,
    p,
    swap,
    size_average,
    reduce,
    reduction,
    test_flags,
    fn_tree,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_inputs
    anchor_dtype, anchor = input_dtype[0], x[0]
    positive_dtype, positive = input_dtype[1], x[1]
    negative_dtype, negative = input_dtype[2], x[2]
    helpers.test_frontend_function(
        input_dtypes=[anchor_dtype, positive_dtype, negative_dtype],
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        anchor=anchor,
        positive=positive,
        negative=negative,
        margin=margin,
        p=p,
        swap=swap,
        size_average=size_average,
        reduce=reduce,
        reduction=reduction,
    )


# multilabel soft margin loss
@handle_frontend_test(
    fn_tree="torch.nn.functional.multilabel_soft_margin_loss",
    dtype_and_inputs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        allow_inf=False,
        shared_dtype=True,
        min_num_dims=1,
    ),
    size_average=st.booleans(),
    reduce=st.booleans(),
    reduction=st.sampled_from(["none", "mean", "sum"]),
    test_with_out=st.just(False),
)
def test_torch_multilabel_soft_margin_loss(
    *,
    dtype_and_inputs,
    size_average,
    reduce,
    reduction,
    test_flags,
    fn_tree,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_inputs
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        target=x[1],
        size_average=size_average,
        reduce=reduce,
        reduction=reduction,
    )


# triplet margin distance loss
@handle_frontend_test(
    fn_tree="torch.nn.functional.triplet_margin_with_distance_loss",
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
    distance_function=st.sampled_from([cosine_similarity, None]),
    margin=st.floats(min_value=-10, max_value=10),
    swap=st.booleans(),
    reduction=st.sampled_from(["none", "mean", "sum"]),
    test_with_out=st.just(False),
)
def test_torch_triplet_margin_with_distance_loss(
    *,
    dtype_and_inputs,
    distance_function,
    margin,
    swap,
    reduction,
    test_flags,
    fn_tree,
    frontend,
    on_device,
):
    input_dtype, x = dtype_and_inputs
    anchor_dtype, anchor = input_dtype[0], x[0]
    positive_dtype, positive = input_dtype[1], x[1]
    negative_dtype, negative = input_dtype[2], x[2]
    test_flags.num_positional_args = len(x)
    helpers.test_frontend_function(
        input_dtypes=[anchor_dtype, positive_dtype, negative_dtype],
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        anchor=anchor,
        positive=positive,
        negative=negative,
        distance_function=distance_function,
        margin=margin,
        swap=swap,
        reduction=reduction,
    )
