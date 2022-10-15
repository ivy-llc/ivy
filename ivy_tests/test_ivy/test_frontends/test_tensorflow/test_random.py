# global
import ivy
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@handle_cmd_line_args
@given(
    dtype_mean_stddev=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True
    ),
    dtype_seed=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer")),
    dtypes=helpers.get_dtypes(
        "float", full=False, key="dtype"),

    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.random.normal"
    ),
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),

)
def test_tensorflow_random_normal(dtype_mean_stddev,
                                  dtype_seed,
                                  dtypes,
                                  num_positional_args,
                                  shape,
                                  ):
    input_mean_stddev_dtype, input_mean_stddev = dtype_mean_stddev
    input_seed_dtype, input_seed_values = dtype_seed

    mean, stddev = input_mean_stddev
    seed = input_seed_values

    helpers.test_frontend_function(
        input_dtypes=input_mean_stddev_dtype + input_seed_dtype,
        as_variable_flags=[False],
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=[False],
        frontend="tensorflow",
        fn_tree="random.normal",
        shape=shape,
        mean=mean,
        stddev=stddev,
        dtype=dtypes[0],
        seed=seed
    )
