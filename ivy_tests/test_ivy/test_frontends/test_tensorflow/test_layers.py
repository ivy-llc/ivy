import numpy as np
from ivy_tests.test_ivy import helpers

# concatenate
num_positional_args = (
    helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.keras.layers.concatenate"
    ),
)
native_array = (helpers.array_bools(),)


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
