# global
from hypothesis import given, strategies as st
import ivy

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.tensorflow as ivy_tf
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


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
    fw
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
        from_logits=from_logits
    )
