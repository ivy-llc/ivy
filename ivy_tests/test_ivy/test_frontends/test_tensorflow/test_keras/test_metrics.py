# global
import numpy as np
from hypothesis import strategies as st
import ivy

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


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
    max_label_val=None,
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
        dtype.append("int32")
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

    return dtype, pred, labels


# binary_accuracy
@handle_frontend_test(
    fn_tree="tensorflow.keras.metrics.binary_accuracy",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
        min_num_dims=1,
    ),
    threshold=st.floats(min_value=0.0, max_value=1.0),
    test_with_out=st.just(False),
)
def test_tensorflow_binary_accuracy(
    *,
    dtype_and_x,
    threshold,
    frontend,
    test_flags,
    fn_tree,
    on_device,
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
        y_true=x[0],
        y_pred=x[1],
        threshold=threshold,
    )


# sparse_categorical_crossentropy
@handle_frontend_test(
    fn_tree="tensorflow.keras.metrics.sparse_categorical_crossentropy",
    y_true=st.lists(st.integers(min_value=0, max_value=4), min_size=1, max_size=1),
    dtype_y_pred=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=(5,),
        min_value=-10,
        max_value=10,
    ),
    from_logits=st.booleans(),
    test_with_out=st.just(False),
)
def test_sparse_categorical_crossentropy(
    *,
    y_true,
    dtype_y_pred,
    from_logits,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    backend_fw,
):
    y_true = ivy.array(y_true, dtype=ivy.int32)
    dtype, y_pred = dtype_y_pred
    y_pred = y_pred[0]
    # Perform softmax on prediction if it's not a probability distribution.
    if not from_logits:
        y_pred = ivy.exp(y_pred) / ivy.sum(ivy.exp(y_pred))

    helpers.test_frontend_function(
        input_dtypes=[ivy.int32] + dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        y_true=y_true,
        y_pred=y_pred[0],
        from_logits=from_logits,
    )


# log_cosh
@handle_frontend_test(
    fn_tree="tensorflow.keras.metrics.log_cosh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=False,
        min_num_dims=1,
        large_abs_safety_factor=2,
        small_abs_safety_factor=2,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_log_cosh(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
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
        y_true=x[0],
        y_pred=x[1],
    )


# binary_crossentropy
@handle_frontend_test(
    fn_tree="tensorflow.keras.metrics.binary_crossentropy",
    dtype_pred_and_labels=_dtype_pred_and_labels(
        available_dtypes=helpers.get_dtypes("float"),
        min_pred_val=1e-6,
        max_label_val=5,
        min_dim_size=1,
        min_num_dims=1,
    ),
    from_logits=st.booleans(),
    label_smoothing=helpers.floats(min_value=0.0, max_value=1.0),
    test_with_out=st.just(False),
)
def test_binary_crossentropy(
    *,
    dtype_pred_and_labels,
    from_logits,
    label_smoothing,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    backend_fw,
):
    input_dtype, y_pred, y_true = dtype_pred_and_labels
    helpers.test_frontend_function(
        input_dtypes=input_dtype[::-1],
        backend_to_test=backend_fw,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-1,
        atol=1e-1,
        y_true=y_true,
        y_pred=y_pred,
        from_logits=from_logits,
        label_smoothing=label_smoothing,
    )


# categorical_crossentropy
@handle_frontend_test(
    fn_tree="tensorflow.keras.metrics.categorical_crossentropy",
    y_true=st.lists(
        st.integers(min_value=0, max_value=4), min_size=1, max_size=1
    ),  # ToDo: we should be using the helpers
    dtype_y_pred=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=(5,),
        min_value=-10,
        max_value=10,
    ),
    from_logits=st.booleans(),
    label_smoothing=helpers.floats(min_value=0.0, max_value=1.0),
    test_with_out=st.just(False),
)
def test_categorical_crossentropy(
    *,
    y_true,
    dtype_y_pred,
    from_logits,
    label_smoothing,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    backend_fw,
):
    y_true = ivy.array(y_true, dtype=ivy.float32)
    dtype, y_pred = dtype_y_pred

    # Perform softmax on prediction if it's not a probability distribution.
    if not from_logits:
        y_pred = ivy.exp(y_pred) / ivy.sum(ivy.exp(y_pred))

    helpers.test_frontend_function(
        input_dtypes=[ivy.float32, dtype],
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        y_true=y_true,
        y_pred=y_pred,
        from_logits=from_logits,
        label_smoothing=label_smoothing,
    )


@st.composite
def _binary_focal_args(draw):
    shape = st.tuples(st.integers(1, 10), st.integers(1, 10), st.integers(1, 10))
    common_float_dtype = helpers.get_dtypes("float", full=False)

    from_logits = draw(
        helpers.dtype_and_values(
            available_dtypes=draw(helpers.get_dtypes("bool")), shape=(1,)
        )
    )

    if from_logits[0]:
        min_value = -10.0
        max_value = 10.0
    else:
        min_value = 0.0
        max_value = 1.0

    dtype_y_true = draw(
        helpers.dtype_and_values(
            available_dtypes=draw(helpers.get_dtypes("integer")),
            min_value=0,
            max_value=2,
            exclude_max=True,
            shape=draw(st.shared(shape, key="shape")),
        )
    )
    dtype_y_pred = draw(
        helpers.dtype_and_values(
            dtype=draw(st.shared(common_float_dtype, key="float_dtype")),
            min_value=min_value,
            max_value=max_value,
            shape=draw(st.shared(shape, key="shape")),
        )
    )
    dtype_label_smoothing = draw(
        helpers.dtype_and_values(
            dtype=draw(st.shared(common_float_dtype, key="float_dtype")),
            min_value=0.0,
            max_value=1.0,
            exclude_min=False,
            exclude_max=False,
            shape=(1,),
        )
    )
    dtype_gamma = draw(
        helpers.dtype_and_values(
            dtype=draw(st.shared(common_float_dtype, key="float_dtype")),
            min_value=0.0,
            max_value=10.0,
            shape=(1,),
        )
    )
    # attr = Tidx:type, default = DT_INT32, allowed = [DT_INT32, DT_INT64] > [Op:Mean]
    dtype_axis = draw(
        helpers.dtype_and_values(
            available_dtypes=[ivy.int32, ivy.int64],
            min_value=-len(draw(st.shared(shape, key="shape"))),
            max_value=len(draw(st.shared(shape, key="shape"))),
            shape=(1,),
        )
    )
    dtype_true, y_true = dtype_y_true
    dtype_pred, y_pred = dtype_y_pred
    dtype_gamma, gamma = dtype_gamma
    dtype_from_logits, from_logits = from_logits
    dtype_label_smoothing, label_smoothing = dtype_label_smoothing
    dtype_axis, axis = dtype_axis
    dtypes = [
        dtype_true[0],
        dtype_pred[0],
        dtype_gamma[0],
        dtype_from_logits[0],
        dtype_label_smoothing[0],
        dtype_axis[0],
    ]
    values = [
        y_true[0],
        y_pred[0],
        gamma[0],
        from_logits[0],
        label_smoothing[0],
        axis[0],
    ]
    return dtypes, values


# binary_focal_crossentropy
@handle_frontend_test(
    fn_tree="tensorflow.keras.metrics.binary_focal_crossentropy",
    binary_focal_args=_binary_focal_args(),
    test_with_out=st.just(False),
)
def test_binary_focal_crossentropy(
    *,
    binary_focal_args,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    backend_fw,
):
    dtypes, values = binary_focal_args
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        y_true=values[0],
        y_pred=values[1],
        gamma=values[2],
        from_logits=values[3],
        label_smoothing=values[4],
        axis=values[5],
    )


# sparse_top_k_categorical_accuracy
@handle_frontend_test(
    fn_tree="tensorflow.keras.metrics.sparse_top_k_categorical_accuracy",
    dtype_pred_and_labels=_dtype_pred_and_labels(
        available_dtypes=helpers.get_dtypes("float"),
        min_pred_val=1e-6,
        max_label_val=5,
        sparse_label=True,
        shape=(5, 10),
    ),
    k=st.integers(min_value=3, max_value=10),
    test_with_out=st.just(False),
)
def test_sparse_top_k_categorical_accuracy(
    *,
    dtype_pred_and_labels,
    k,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    backend_fw,
):
    input_dtype, y_pred, y_true = dtype_pred_and_labels
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        y_true=y_true,
        y_pred=y_pred,
        k=k,
    )


# categorical_accuracy
@handle_frontend_test(
    fn_tree="tensorflow.keras.metrics.categorical_accuracy",
    dtype_and_y=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
        shape=helpers.get_shape(
            allow_none=False,
            min_num_dims=1,
        ),
    ),
    test_with_out=st.just(False),
)
def test_categorical_accuracy(
    *,
    dtype_and_y,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    backend_fw,
):
    input_dtype, y = dtype_and_y
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        y_true=y[0],
        y_pred=y[1],
    )


# kl_divergence
@handle_frontend_test(
    fn_tree="tensorflow.keras.metrics.kl_divergence",
    aliases=["tensorflow.keras.metrics.kld"],
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
        min_num_dims=1,
    ),
)
def test_tensorflow_kl_divergence(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
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
        y_true=x[0],
        y_pred=x[1],
    )


# mean_absolute_error
@handle_frontend_test(
    fn_tree="tensorflow.keras.metrics.mean_absolute_error",
    aliases=["tensorflow.keras.metrics.mae"],
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
        min_num_dims=1,
        large_abs_safety_factor=2,
        small_abs_safety_factor=2,
        safety_factor_scale="log",
    ),
)
def test_tensorflow_mean_absolute_error(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
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
        y_true=x[0],
        y_pred=x[1],
    )


# poisson
@handle_frontend_test(
    fn_tree="tensorflow.keras.metrics.poisson",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
        min_num_dims=1,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_poisson(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
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
        y_true=x[0],
        y_pred=x[1],
    )


# mean_squared_error
@handle_frontend_test(
    fn_tree="tensorflow.keras.metrics.mean_squared_error",
    aliases=["tensorflow.keras.metrics.mse"],
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
        min_num_dims=1,
        large_abs_safety_factor=2,
        small_abs_safety_factor=2,
    ),
)
def test_tensorflow_mean_squared_error(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
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
        y_true=x[0],
        y_pred=x[1],
    )


# mean_absolute_percentage_error
@handle_frontend_test(
    fn_tree="tensorflow.keras.metrics.mean_absolute_percentage_error",
    aliases=["tensorflow.keras.metrics.mape"],
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
        min_num_dims=1,
        large_abs_safety_factor=2,
        small_abs_safety_factor=2,
    ),
)
def test_tensorflow_mean_absolute_percentage_error(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
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
        y_true=x[0],
        y_pred=x[1],
    )


# hinge
@handle_frontend_test(
    fn_tree="tensorflow.keras.metrics.hinge",
    dtype_pred_and_labels=_dtype_pred_and_labels(
        available_dtypes=helpers.get_dtypes("float"),
        label_set=[-1, 1],
        min_num_dims=2,
        min_dim_size=2,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_hinge(
    *,
    dtype_pred_and_labels,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    backend_fw,
):
    input_dtype, y_pred, y_true = dtype_pred_and_labels
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        y_pred=y_pred,
        y_true=y_true,
    )


# squared_hinge
@handle_frontend_test(
    fn_tree="tensorflow.keras.metrics.squared_hinge",
    dtype_pred_and_labels=_dtype_pred_and_labels(
        available_dtypes=helpers.get_dtypes("float"),
        label_set=[-1, 1],
        min_num_dims=2,
        min_dim_size=2,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_squared_hinge(
    *,
    dtype_pred_and_labels,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    backend_fw,
):
    input_dtype, y_pred, y_true = dtype_pred_and_labels
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        y_pred=y_pred,
        y_true=y_true,
    )


# mean_squared_logarithmic_error
@handle_frontend_test(
    fn_tree="tensorflow.keras.metrics.mean_squared_logarithmic_error",
    aliases=["tensorflow.keras.metrics.msle"],
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_num_dims=1,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_metrics_mean_squared_logarithmic_error(
    *,
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
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
        y_true=x[0],
        y_pred=x[1],
    )


# Cosine Similarity
@handle_frontend_test(
    fn_tree="tensorflow.keras.metrics.cosine_similarity",
    d_type=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"), shared_dtype=True, num_arrays=2
    ),
    y_true=helpers.array_values(
        dtype=ivy.int32, shape=(1, 5), min_value=1, max_value=5
    ),
    y_pred=helpers.array_values(
        dtype=ivy.int32, shape=(1, 5), min_value=5, max_value=10
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_cosine_similarity(
    *,
    d_type,
    y_true,
    y_pred,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    backend_fw,
):
    helpers.test_frontend_function(
        input_dtypes=d_type[0],
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        y_true=y_true,
        y_pred=y_pred,
    )
