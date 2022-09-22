# global
import numpy as np
from hypothesis import given, strategies as st
import ivy

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@st.composite
def _dtype_pred_and_labels(
    draw,
    *,
    dtype=None,
    available_dtypes=helpers.get_dtypes("numeric"),
    shared_dtype=False,
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
            helpers.list_of_length(
                x=st.integers(min_value=0, max_value=len(label_set) - 1), length=length
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
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
        min_num_dims=1,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.binary_accuracy"
    ),
    threshold=st.floats(min_value=0.0, max_value=1.0),
)
def test_tensorflow_binary_accuracy(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    fw,
    threshold,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="keras.metrics.binary_accuracy",
        y_true=np.asarray(x[0], dtype=input_dtype[0]),
        y_pred=np.asarray(x[1], dtype=input_dtype[1]),
        threshold=threshold,
    )


# sparse_categorical_crossentropy
@handle_cmd_line_args
@given(
    y_true=st.lists(st.integers(min_value=0, max_value=4), min_size=1, max_size=1),
    dtype_y_pred=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=(5,),
        min_value=-10,
        max_value=10,
    ),
    from_logits=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.sparse_categorical_crossentropy"
    ),
)
def test_sparse_categorical_crossentropy(
    y_true,
    dtype_y_pred,
    from_logits,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    y_true = ivy.array(y_true, dtype=ivy.int32)
    dtype, y_pred = dtype_y_pred

    # Perform softmax on prediction if it's not a probability distribution.
    if not from_logits:
        y_pred = ivy.exp(y_pred) / ivy.sum(ivy.exp(y_pred))

    helpers.test_frontend_function(
        input_dtypes=[ivy.int32, dtype],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="keras.metrics.sparse_categorical_crossentropy",
        y_true=y_true,
        y_pred=y_pred,
        from_logits=from_logits,
    )


# mean_absolute_error
@handle_cmd_line_args
@given(
    input_dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shared_dtype=True,
        num_arrays=2,
        min_num_dims=1,
        large_abs_safety_factor=2,
        small_abs_safety_factor=2,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.mean_absolute_error"
    ),
)
def test_tensorflow_mean_absolute_error(
    input_dtype_x, as_variable, num_positional_args, native_array, fw
):
    input_dtype, x = input_dtype_x

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="keras.metrics.mean_absolute_error",
        y_true=np.asarray(x[0], dtype=input_dtype[0]),
        y_pred=np.asarray(x[1], dtype=input_dtype[1]),
    )


# binary_crossentropy
@handle_cmd_line_args
@given(
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
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.binary_crossentropy"
    ),
)
def test_binary_crossentropy(
    y_true,
    dtype_y_pred,
    from_logits,
    label_smoothing,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    y_true = ivy.array(y_true, dtype=ivy.int32)
    dtype, y_pred = dtype_y_pred

    # Perform softmax on prediction if it's not a probability distribution.
    if not from_logits:
        y_pred = ivy.exp(y_pred) / ivy.sum(ivy.exp(y_pred))

    helpers.test_frontend_function(
        input_dtypes=[ivy.int32, dtype],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="keras.metrics.binary_crossentropy",
        y_true=y_true,
        y_pred=y_pred,
        from_logits=from_logits,
        label_smoothing=label_smoothing,
    )


# sparse_top_k_categorical_accuracy
@handle_cmd_line_args
@given(
    dtype_pred_and_labels=_dtype_pred_and_labels(
        available_dtypes=helpers.get_dtypes("float"),
        min_pred_val=1e-6,
        max_label_val=5,
        sparse_label=True,
        shape=(5, 10),
    ),
    k=st.integers(min_value=3, max_value=10),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.sparse_top_k_categorical_accuracy"
    ),
)
def test_sparse_top_k_categorical_accuracy(
    dtype_pred_and_labels, k, as_variable, num_positional_args, native_array, fw
):
    input_dtype, y_pred, y_true = dtype_pred_and_labels
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="keras.metrics.sparse_top_k_categorical_accuracy",
        y_true=y_true,
        y_pred=y_pred,
        k=k,
    )


# categorical_accuracy
@handle_cmd_line_args
@given(
    dtype_and_y=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
        shape=helpers.get_shape(
            allow_none=False,
            min_num_dims=1,
        ),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.categorical_accuracy"
    ),
)
def test_categorical_accuracy(
    dtype_and_y, as_variable, num_positional_args, native_array, fw
):
    input_dtype, y = dtype_and_y
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="keras.metrics.categorical_accuracy",
        y_true=y[0],
        y_pred=y[1],
    )


# kl_divergence
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
        min_num_dims=1,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.kl_divergence"
    ),
)
def test_tensorflow_kl_divergence(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="keras.metrics.kl_divergence",
        y_true=np.asarray(x[0], dtype=input_dtype[0]),
        y_pred=np.asarray(x[1], dtype=input_dtype[1]),
    )


# poisson
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
        min_num_dims=1,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.poisson"
    ),
)
def test_tensorflow_poisson(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="keras.metrics.poisson",
        y_true=np.asarray(x[0], dtype=input_dtype[0]),
        y_pred=np.asarray(x[1], dtype=input_dtype[1]),
    )


# mean_squared_error
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
        min_num_dims=1,
        large_abs_safety_factor=2,
        small_abs_safety_factor=2,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.mean_squared_error"
    ),
)
def test_tensorflow_mean_squared_error(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="keras.metrics.mean_squared_error",
        y_true=np.asarray(x[0], dtype=input_dtype[0]),
        y_pred=np.asarray(x[1], dtype=input_dtype[1]),
    )


# mean_absolute_percentage_error
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
        min_num_dims=1,
        large_abs_safety_factor=2,
        small_abs_safety_factor=2,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.mean_absolute_percentage_error"
    ),
)
def test_tensorflow_mean_absolute_percentage_error(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="keras.metrics.mean_absolute_percentage_error",
        y_true=np.asarray(x[0], dtype=input_dtype[0]),
        y_pred=np.asarray(x[1], dtype=input_dtype[1]),
    )


# hinge
@handle_cmd_line_args
@given(
    dtype_pred_and_labels=_dtype_pred_and_labels(
        available_dtypes=helpers.get_dtypes("float"),
        label_set=[-1, 1],
        min_num_dims=2,
        min_dim_size=2,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.hinge"
    ),
)
def test_tensorflow_hinge(
    dtype_pred_and_labels, as_variable, num_positional_args, native_array, fw
):
    input_dtype, y_pred, y_true = dtype_pred_and_labels
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="keras.metrics.hinge",
        y_pred=np.asarray(y_pred, dtype=input_dtype[0]),
        y_true=np.asarray(y_true, dtype=input_dtype[1]),
    )


# squared_hinge
@handle_cmd_line_args
@given(
    dtype_pred_and_labels=_dtype_pred_and_labels(
        available_dtypes=helpers.get_dtypes("float"),
        label_set=[-1, 1],
        min_num_dims=2,
        min_dim_size=2,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.squared_hinge"
    ),
)
def test_tensorflow_squared_hinge(
    dtype_pred_and_labels, as_variable, num_positional_args, native_array, fw
):
    input_dtype, y_pred, y_true = dtype_pred_and_labels
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="keras.metrics.squared_hinge",
        y_pred=np.asarray(y_pred, dtype=input_dtype[0]),
        y_true=np.asarray(y_true, dtype=input_dtype[1]),
    )


# mean_squared_logarithmic_error
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_num_dims=1,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.mean_squared_logarithmic_error"  # noqa: E501
    ),
)
def test_tensorflow_metrics_mean_squared_logarithmic_error(
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
        frontend="tensorflow",
        fn_tree="keras.metrics.mean_squared_logarithmic_error",
        y_true=np.asarray(x[0], dtype=input_dtype[0]),
        y_pred=np.asarray(x[1], dtype=input_dtype[1]),
    )
