# from pydoc import Helper
from hypothesis import given
import numpy as np
from ivy_tests.test_ivy import helpers
# from .test_raw_ops import _arrays_idx_n_dtypes

# layer_concatenate
given( 
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        shared_dtype=True,
    ),  
      # xs_n_input_dtypes_n_unique_idx=_arrays_idx_n_dtypes(),
      # as_variable=Helper.array_bools(),

    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.keras.layers.concatenate"
    ),
    native_array=helpers.array_bools(),
)


def test_tensorflow_keras_layers_concatenate(
    xs_n_input_dtypes_n_unique_idx, 
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
        fn_name="concatenate",
        values=xs,
        axis=unique_idx,   
    )
     
