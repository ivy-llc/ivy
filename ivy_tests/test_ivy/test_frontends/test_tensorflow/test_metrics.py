# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.tensorflow as ivy_tf


@given(
    y_true=st.lists(st.integers(min_value=0, max_value=5), min_size=1, max_size=1),
    y_pred=helpers.dtype_and_values(
        available_dtypes=ivy_tf.valid_float_dtypes,
        shape=(5,),
        min_value=-10,
        max_value=10,
    ),
    from_logits=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.sparse_categorical_crossentropy"
    )
)
def test_sparse_categorical_crossentropy(
        y_true,
        y_pred,
        from_logits,
        num_positional_args,
        fw
):
    y_true = np.array(y_true, dtype='int32')
    dtype = y_pred[0]
    y_pred = np.array(y_pred[1], dtype=dtype)

    # Perform softmax on prediction if it's not a probability distribution.
    if not from_logits:
        y_pred = np.exp(y_pred) / np.sum(np.exp(y_pred))

    helpers.test_frontend_function(
        input_dtypes=['int32', dtype],
        as_variable_flags=False,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=True,
        fw=fw,
        frontend="tensorflow",
        fn_tree="metrics.sparse_categorical_crossentropy",
        y_true=y_true,
        y_pred=y_pred,
        from_logits=from_logits
    )
