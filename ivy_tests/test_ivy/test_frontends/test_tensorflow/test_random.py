from hypothesis import given

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@handle_cmd_line_args
@given(
    dtype_minval_maxval=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"), num_arrays=2, shared_dtype=True
    ),
    dtype_seed=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("integer")),
    dtypes=helpers.get_dtypes("float", full=False),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.random.uniform"
    ),
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
)
def test_tensorflow_random_uniform(
    dtype_minval_maxval,
    dtype_seed,
    dtypes,
    num_positional_args,
    shape,
    as_variable,
    native_array,
    fw,
    with_out,
):
    input_minval_maxval_dtype, (minval, maxval) = dtype_minval_maxval
    input_seed_dtype, seed = dtype_seed

    helpers.test_frontend_function(
        input_dtypes=input_minval_maxval_dtype + input_seed_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="random.uniform",
        shape=shape,
        minval=minval[0],
        maxval=maxval[0],
        dtype=dtypes[0],
        seed=seed,
    )
