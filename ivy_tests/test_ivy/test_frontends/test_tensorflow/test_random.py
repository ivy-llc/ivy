from hypothesis import given

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@handle_cmd_line_args
@given(
    dtype_and_min=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_value=-1000,
        max_value=100,
    ),
    dtype_and_max=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_value=101,
        max_value=1000,
    ),
    dtype_and_seed=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        max_num_dims=1,
    ),
    dtypes=helpers.get_dtypes(kind="numeric", full=False, key="dtypes"),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.random.uniform"
    ),
    shape=helpers.get_shape(),
)
def test_tensorflow_random_uniform(
    dtype_and_min,
    dtype_and_max,
    dtype_and_seed,
    dtypes,
    num_positional_args,
    shape,
    as_variable,
    native_array,
    with_out,
):
    min_dtype, minval = dtype_and_min
    max_dtype, maxval = dtype_and_max
    seed_dtype, seed = dtype_and_seed

    helpers.test_frontend_function(
        input_dtypes=min_dtype + max_dtype + seed_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="random.uniform",
        shape=shape,
        minval=minval[0],
        maxval=maxval[0],
        seed=seed[0],
        dtype="numeric",
    )
