# global
import numpy as np
from hypothesis import given, strategies as st
import ivy

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.tensorflow as ivy_tf
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@st.composite
def _get_dtype_y_true_y_pred(
    draw,
    *,
    min_value=None,
    max_value=None,
    large_value_safety_factor=1.1,
    small_value_safety_factor=1.1,
    allow_inf=False,
    exclude_min=False,
    exclude_max=False,
    min_num_dims=0,
    max_num_dims=5,
    min_dim_size=1,
    max_dim_size=10,
    shape=None,
    ret_shape=False,
):
    if isinstance(min_dim_size, st._internal.SearchStrategy):
        min_dim_size = draw(min_dim_size)
    if isinstance(max_dim_size, st._internal.SearchStrategy):
        max_dim_size = draw(max_dim_size)

    dtype_true = draw(
        helpers.array_dtypes(
            num_arrays=1,
            available_dtypes=ivy_tf.valid_int_dtypes,
        )
    )[0]
    dtype_pred = draw(
        helpers.array_dtypes(
            num_arrays=1,
            available_dtypes=ivy_tf.valid_float_dtypes,
        )
    )[0]

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
    y_true = draw(
        helpers.array_values(
            dtype=dtype_true,
            shape=shape,
            min_value=min_value,
            max_value=max_value,
            allow_inf=allow_inf,
            exclude_min=exclude_min,
            exclude_max=exclude_max,
            large_value_safety_factor=large_value_safety_factor,
            small_value_safety_factor=small_value_safety_factor,
        )
    )

    y_pred = draw(
        helpers.array_values(
            dtype=dtype_pred,
            shape=shape,
            min_value=min_value,
            max_value=max_value,
            allow_inf=allow_inf,
            exclude_min=exclude_min,
            exclude_max=exclude_max,
            large_value_safety_factor=large_value_safety_factor,
            small_value_safety_factor=small_value_safety_factor,
        )
    )

    if ret_shape:
        return [dtype_pred, dtype_pred], [y_true, y_pred], shape
    return [dtype_pred, dtype_pred], [y_true, y_pred]


# binary_accuracy
@handle_cmd_line_args
@given(
    dtype_y_true_y_pred=_get_dtype_y_true_y_pred(
        min_num_dims=1,
        max_num_dims=5,
        min_value=0,
        max_value=1,
    ),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.binary_accuracy"
    ),
    native_array=st.booleans(),
)
def test_tensorflow_binary_accuracy(
    dtype_y_true_y_pred, as_variable, num_positional_args, native_array, fw
):
    input_dtype, y_true_y_pred = dtype_y_true_y_pred
    y_true, y_pred = y_true_y_pred
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="keras.metrics.binary_accuracy",
        y_true=np.asarray(y_true, dtype=input_dtype[0]),
        y_pred=np.asarray(y_pred, dtype=input_dtype[1]),
        threshold=0.5,
    )


# sparse_categorical_crossentropy
@handle_cmd_line_args
@given(
    y_true=st.lists(st.integers(min_value=0, max_value=4), min_size=1, max_size=1),
    dtype_y_pred=helpers.dtype_and_values(
        available_dtypes=ivy_tf.valid_float_dtypes,
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
