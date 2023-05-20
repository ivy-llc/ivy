# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test
import numpy as np


# cross_entropy
@handle_test(
    fn_tree="functional.ivy.cross_entropy",
    dtype_and_true=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        min_value=1e-04,
        max_value=1,
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
    ),
    dtype_and_pred=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=1e-04,
        max_value=1,
        allow_inf=False,
        exclude_min=True,
        exclude_max=True,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
    ),
    reduction=st.sampled_from(["none", "sum", "mean"]),
    axis=helpers.ints(min_value=-1, max_value=0),
    epsilon=helpers.floats(min_value=0, max_value=0.49),
)
def test_cross_entropy(
    dtype_and_true,
    dtype_and_pred,
    reduction,
    axis,
    epsilon,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    pred_dtype, pred = dtype_and_pred
    true_dtype, true = dtype_and_true

    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=true_dtype + pred_dtype,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-02,
        atol_=1e-02,
        true=true[0],
        pred=pred[0],
        axis=axis,
        epsilon=epsilon,
        reduction=reduction,
    )


@st.composite
def _dtype_pred_and_labels(
    draw,
    *,
    dtype=None,
    available_dtypes=helpers.get_dtypes("numeric"),
    min_pred_val=0,
    max_pred_val=1,  # predication array output as probabilities
    label_set=None,
    min_label_val=0,
    max_label_val=10,
    allow_inf=False,
    allow_nan=False,
    exclude_min=False,
    exclude_max=False,
    sparse_label=False,
    min_num_dims=0,
    max_num_dims=5,
    min_dim_size=1,
    max_dim_size=10,
    shape=None,
):
    if isinstance(min_dim_size, st._internal.SearchStrategy):
        min_dim_size = draw(min_dim_size)
    if isinstance(max_dim_size, st._internal.SearchStrategy):
        max_dim_size = draw(max_dim_size)
    if isinstance(available_dtypes, st._internal.SearchStrategy):
        available_dtypes = draw(available_dtypes)

    if dtype is None:
        assert available_dtypes is not None, "Unspecified dtype or available_dtypes."
        dtype = draw(
            helpers.array_dtypes(
                num_arrays=1,
                available_dtypes=available_dtypes,
            )
        )
        dtype.append("float32")

    # initialize shapes for pred and label
    if shape is not None:
        if not isinstance(shape, (tuple, list)):
            shape = draw(shape)
    else:
        shape = draw(
            st.shared(
                helpers.get_shape(
                    min_num_dims=min_num_dims,
                    max_num_dims=max_num_dims,
                    min_dim_size=min_dim_size,
                    max_dim_size=max_dim_size,
                ),
                key="shape",
            )
        )

    if not sparse_label:
        label_shape = shape
    else:
        label_shape = shape[:-1]

    pred = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=shape,
            min_value=min_pred_val,
            max_value=max_pred_val,
            allow_inf=allow_inf,
            allow_nan=allow_nan,
            exclude_min=exclude_min,
            exclude_max=exclude_max,
        )
    )

    num_classes = (
        pred.shape[1] if len(pred.shape) > 1 else 1
    )  # The number of classes corresponds to the second dimension of the pred array
    pos_weights = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=(
                num_classes,
            ),  # The shape of pos_weights should be a vector with length num_classes
            min_value=min_pred_val,
            max_value=max_pred_val,
            allow_inf=allow_inf,
            allow_nan=allow_nan,
            exclude_min=exclude_min,
            exclude_max=exclude_max,
        )
    )

    # generate labels by restriction
    if label_set is not None:
        length = 1
        for _ in label_shape:
            length *= _
        indices = draw(
            helpers.list_of_size(
                x=st.integers(min_value=0, max_value=len(label_set) - 1),
                size=length,
            )
        )
        values = [label_set[_] for _ in indices]
        array = np.array(values)
        labels = array.reshape(label_shape).tolist()
    else:
        labels = draw(
            helpers.array_values(
                dtype=dtype[1],
                shape=label_shape,
                min_value=min_label_val,
                max_value=max_label_val,
                allow_inf=allow_inf,
                allow_nan=allow_nan,
                exclude_min=exclude_min,
                exclude_max=exclude_max,
            )
        )

    return dtype, pred, labels, pos_weights


@handle_test(
    fn_tree="functional.ivy.binary_cross_entropy",
    dtype_pred_and_labels=_dtype_pred_and_labels(
        available_dtypes=helpers.get_dtypes("float"),
        min_pred_val=1e-6,
        max_label_val=5,
        min_dim_size=1,
        min_num_dims=1,
    ),
    from_logits=st.just(True),
    # from_logits=st.booleans(),
    epsilon=helpers.floats(min_value=0.0, max_value=1.0),
    reduction=st.sampled_from(["none", "sum", "mean"]),
    axis=helpers.ints(min_value=-1, max_value=0),
    test_with_out=st.just(False),
)
def test_binary_cross_entropy(
    dtype_pred_and_labels,
    from_logits,
    reduction,
    axis,
    epsilon,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtypes, true, pred, pos_weight = dtype_pred_and_labels

    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtypes,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        true=true[0],
        pred=pred[0],
        axis=axis,
        epsilon=epsilon,
        reduction=reduction,
        from_logits=from_logits,
        pos_weight=pos_weight[0],
    )


# @st.composite
# def _binary_cross_entropy_args(draw):
#     common_float_dtype = helpers.get_dtypes("float", full=False)

#     from_logits = draw(
#         helpers.dtype_and_values(
#             available_dtypes=draw(helpers.get_dtypes("bool")), shape=(1,)
#         )
#     )

#     if from_logits[0]:
#         min_value = -10.0
#         max_value = 10.0
#         dtype_pos_weight = draw(
#             helpers.dtype_and_values(
#                 dtype=draw(st.shared(common_float_dtype, key="float_dtype")),
#                 min_value=min_value,
#                 max_value=max_value,
#                 shape=shared_shape,
#             )
#         )
#     else:
#         min_value = 0.0
#         max_value = 1.0
#         dtype_pos_weight = None

#     shape = st.tuples(st.integers(1, 10), st.integers(1, 10), st.integers(1, 10))
#     shared_shape = draw(st.shared(shape, key="shape"))

#     dtype_y_true = draw(
#         helpers.dtype_and_values(
#             available_dtypes=draw(helpers.get_dtypes("integer")),
#             min_value=0,
#             max_value=2,
#             exclude_max=True,
#             shape=shared_shape,
#         )
#     )
#     dtype_y_pred = draw(
#         helpers.dtype_and_values(
#             dtype=draw(st.shared(common_float_dtype, key="float_dtype")),
#             min_value=min_value,
#             max_value=max_value,
#             shape=shared_shape,
#         )
#     )
#     # dtype_pos_weight = draw(
#     #     helpers.dtype_and_values(
#     #         dtype=draw(st.shared(common_float_dtype, key="float_dtype")),
#     #         min_value=min_value,
#     #         max_value=max_value,
#     #         shape=shared_shape,
#     #     )
#     # )
#     dtype_label_smoothing = draw(
#         helpers.dtype_and_values(
#             dtype=draw(st.shared(common_float_dtype, key="float_dtype")),
#             min_value=0.0,
#             max_value=1.0,
#             exclude_min=False,
#             exclude_max=False,
#             shape=(1,),
#         )
#     )
#     # attr = Tidx:type, default = DT_INT32, allowed = [DT_INT32, DT_INT64] > [Op:Mean]
#     dtype_axis = draw(
#         helpers.dtype_and_values(
#             available_dtypes=[ivy.int32, ivy.int64],
#             min_value=-len(shared_shape),
#             max_value=len(shared_shape),
#             shape=(1,),
#         )
#     )

#     dtype_true, y_true = dtype_y_true
#     dtype_pred, y_pred = dtype_y_pred
#     dtype_posweight, pos_weight = dtype_pos_weight
#     dtype_from_logits, from_logits = from_logits
#     dtype_label_smoothing, label_smoothing = dtype_label_smoothing
#     dtype_axis, axis = dtype_axis
#     dtypes = [
#         dtype_true[0],
#         dtype_pred[0],
#         dtype_posweight[0],
#         dtype_from_logits[0],
#         dtype_label_smoothing[0],
#         dtype_axis[0],
#     ]
#     values = [
#         y_true[0],
#         y_pred[0],
#         pos_weight[0],
#         from_logits[0],
#         label_smoothing[0],
#         axis[0],
#     ]
#     return dtypes, values
# # binary_cross_entropy
# @handle_test(
#     fn_tree="functional.ivy.binary_cross_entropy",
#     dtypes_and_val= _binary_cross_entropy_args(),
#     reduction=st.sampled_from(["none", "sum", "mean"]),
# )
# def test_binary_cross_entropy(
#     dtypes_and_val,
#     reduction,
#     test_flags,
#     backend_fw,
#     fn_name,
#     on_device,
#     ground_truth_backend,
# ):
#     dtypes, val = dtypes_and_val
#     helpers.test_function(
#         ground_truth_backend=ground_truth_backend,
#         input_dtypes=dtypes,
#         test_flags=test_flags,
#         fw=backend_fw,
#         fn_name=fn_name,
#         on_device=on_device,
#         rtol_=1e-1,
#         atol_=1e-1,
#         true=val[0],
#         pred=val[1],
#         pos_weight=val[2],
#         from_logits=val[3],
#         epsilon=val[4],
#         axis=val[5],
#         reduction=reduction,
#     )
# @st.composite
# def _binary_cross_entropy_args(draw):
#     common_float_dtype = helpers.get_dtypes("float", full=False)
#     shape = st.tuples(st.integers(1, 10), st.integers(1, 10), st.integers(1, 10))
#     shared_shape = draw(st.shared(shape, key="shape"))


#     from_logits = draw(
#         helpers.dtype_and_values(
#             available_dtypes=draw(helpers.get_dtypes("bool")), shape=(1,)
#         )
#     )

#     if from_logits[0]:
#         min_value = -10.0
#         max_value = 10.0
#         dtype_pos_weight = draw(
#         helpers.dtype_and_values(
#             dtype=draw(st.shared(common_float_dtype, key="float_dtype")),
#             min_value=min_value,
#             max_value=max_value,
#             shape=shared_shape,
#         )
#     )


#     else:
#         min_value = 0.0
#         max_value = 1.0
#         dtype_pos_weight = None


#     dtype_y_true = draw(
#         helpers.dtype_and_values(
#             dtype=draw(st.shared(common_float_dtype, key="float_dtype")),
#             min_value=0,
#             max_value=2,
#             exclude_max=True,
#             shape=shared_shape,
#         )
#     )
#     dtype_y_pred = draw(
#         helpers.dtype_and_values(
#             dtype=draw(st.shared(common_float_dtype, key="float_dtype")),
#             min_value=min_value,
#             max_value=max_value,
#             shape=shared_shape,
#         )
#     )
#     dtype_label_smoothing = draw(
#         helpers.dtype_and_values(
#             dtype=draw(st.shared(common_float_dtype, key="float_dtype")),
#             min_value=0.0,
#             max_value=1.0,
#             exclude_min=False,
#             exclude_max=False,
#             shape=(1,),
#         )
#     )

#     dtype_axis = draw(
#         helpers.dtype_and_values(
#             available_dtypes=[ivy.int32, ivy.int64],
#             min_value=-len(shared_shape),
#             max_value=len(shared_shape),
#             shape=(1,),
#         )
#     )

#     dtype_true, y_true = dtype_y_true
#     dtype_pred, y_pred = dtype_y_pred

#     # Check if dtype_pos_weight is not None
#     if from_logits[0]:
#         dtype_posweight, pos_weight = dtype_pos_weight
#     else:
#         dtype_posweight, pos_weight = ivy.int32, None

#     dtype_from_logits, from_logits = from_logits
#     dtype_label_smoothing, label_smoothing = dtype_label_smoothing
#     dtype_axis, axis = dtype_axis
#     dtypes = [
#         dtype_true[0],
#         dtype_pred[0],
#         dtype_posweight[0] ,
#         dtype_from_logits[0],
#         dtype_label_smoothing[0],
#         dtype_axis[0],
#     ]
#     values = [
#         y_true[0],
#         y_pred[0],
#         pos_weight[0],
#         from_logits[0],
#         label_smoothing[0],
#         axis[0],
#     ]
#     return dtypes, values


# @handle_test(
#     fn_tree="functional.ivy.binary_cross_entropy",
#     dtypes_and_val= _binary_cross_entropy_args(),
#     reduction=st.sampled_from(["none", "sum", "mean"]),
#     test_with_out=st.just(False),
# )
# def test_binary_cross_entropy(
#     dtypes_and_val,
#     reduction,
#     test_flags,
#     backend_fw,
#     fn_name,
#     on_device,
#     ground_truth_backend,
# ):
#     dtypes, val = dtypes_and_val

#     kwargs = {
#         'true':val[0] ,
#         'pred':val[1] ,
#         'from_logits':val[3],
#         'epsilon': val[4],
#         'axis':  val[5],
#         'reduction': reduction,
#     }

#     if val[3]: # from_logits is True
#         kwargs['pos_weight'] = val[2]
#     else:
#         kwargs['pos_weight'] = None


#     helpers.test_function(
#         ground_truth_backend=ground_truth_backend,
#         input_dtypes=dtypes,
#         test_flags=test_flags,
#         fw=backend_fw,
#         fn_name=fn_name,
#         on_device=on_device,
#         rtol_=1e-1,
#         atol_=1e-1,
#         **kwargs
#     )


# sparse_cross_entropy
@handle_test(
    fn_tree="functional.ivy.sparse_cross_entropy",
    dtype_and_true=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        min_value=0,
        max_value=2,
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=3,
    ),
    dtype_and_pred=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        small_abs_safety_factor=4,
        safety_factor_scale="log",
        max_value=1,
        allow_inf=False,
        exclude_min=True,
        exclude_max=True,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=3,
    ),
    reduction=st.sampled_from(["none", "sum", "mean"]),
    axis=helpers.ints(min_value=-1, max_value=0),
    epsilon=helpers.floats(min_value=0.01, max_value=0.49),
)
def test_sparse_cross_entropy(
    dtype_and_true,
    dtype_and_pred,
    reduction,
    axis,
    epsilon,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    true_dtype, true = dtype_and_true
    pred_dtype, pred = dtype_and_pred
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=true_dtype + pred_dtype,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        true=true[0],
        pred=pred[0],
        axis=axis,
        epsilon=epsilon,
        reduction=reduction,
    )
