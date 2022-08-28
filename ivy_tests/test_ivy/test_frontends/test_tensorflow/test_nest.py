# global
import ivy
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.tensorflow as ivy_tf
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy.valid_dtypes).intersection(set(ivy_tf.valid_dtypes))
        ),
    ),
    expand_composite=st.booleans(),
    use_array=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.nest.flatten"
    ),
)
def test_tensorflow_flatten(
    dtype_and_x,
    expand_composite,
    use_array,
    num_positional_args,
    as_variable,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=[input_dtype],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="nest.flatten",
        structure=np.array(x, dtype=input_dtype) if use_array else x,
        expand_composites=expand_composite,
    )
