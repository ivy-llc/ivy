import ivy.functional.frontends.tensorflow as Tensor
import numpy as np
from ivy_tests.test_ivy import helpers
from .test_raw_ops import _arrays_idx_n_dtypes
from ivy_tests.test_ivy.helpers import handle_frontend_test


# layer_concatenate
@handle_frontend_test(
    xs_n_input_dtypes_n_unique_idx=_arrays_idx_n_dtypes(),
    fn_tree="tensorflow.keras.layers.concatenate",
    available_dtypes=helpers.get_dtypes("valid"),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.keras.layers.concatenate"
    ),
    native_array=helpers.array_bools(),
)
def test_tensorflow_keras_layers_concatenate(
    xs_n_input_dtypes_n_unique_idx,
    fn_tree,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    xs, input_dtypes, unique_idx = xs_n_input_dtypes_n_unique_idx
    xs = [np.asarray(x, dtype=dt) for x, dt in zip(xs, input_dtypes)]
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        frontend_class=Tensor,
        fn_tree=fn_tree,
        values=xs,
        axis=unique_idx,
    )
