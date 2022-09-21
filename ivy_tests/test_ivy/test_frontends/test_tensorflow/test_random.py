# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        allow_inf=False,
        min_num_dims=1,
        min_dim_size=2,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.shuffle"
    ),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_tensorflow_shuffle(
    dtype_and_x, seed, as_variable, num_positional_args, native_array, fw
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        test_values=False,
        fw=fw,
        frontend="tensorflow",
        fn_tree="random.shuffle",
        value=np.asarray(x, dtype=input_dtype),
        seed=seed,
    )
