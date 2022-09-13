# global
import numpy as np
from hypothesis import given

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# shape
@handle_cmd_line_args
@given(
    xs_n_input_dtypes_n_unique_idx=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric")
    ),
    as_variable=helpers.array_bools(num_arrays=1),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.shape"
    ),
    native_array=helpers.array_bools(num_arrays=1),
)
def test_numpy_shape(
    xs_n_input_dtypes_n_unique_idx,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtypes, xs = xs_n_input_dtypes_n_unique_idx
    xs = np.asarray(xs, dtype=input_dtypes)
    ret, ret_gt = helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        with_out=False,
        fw=fw,
        frontend="numpy",
        fn_tree="shape",
        array=xs,
        test_values=False,
    )
    # Manually compare the shape here because ivy.shape doesn't return an array, so
    # ivy.to_numpy will narrow the bit-width, resulting in different dtypes. This is
    # not an issue with the front-end function, but how the testing framework converts
    # non-array function outputs to arrays.
    assert len(ret[0]) == len(ret_gt[0])
    for i, j in zip(ret[0], ret_gt[0]):
        assert i == j
