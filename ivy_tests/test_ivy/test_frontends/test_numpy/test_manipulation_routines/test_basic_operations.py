# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# shape
@handle_frontend_test(
    fn_tree="numpy.shape",
    xs_n_input_dtypes_n_unique_idx=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric")
    ),
)
def test_numpy_shape(
    *,
    xs_n_input_dtypes_n_unique_idx,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtypes, xs = xs_n_input_dtypes_n_unique_idx
    ret, ret_gt = helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        array=xs[0],
    )
    # Manually compare the shape here because ivy.shape doesn't return an array, so
    # ivy.to_numpy will narrow the bit-width, resulting in different dtypes. This is
    # not an issue with the front-end function, but how the testing framework converts
    # non-array function outputs to arrays.
    assert len(ret[0]) == len(ret_gt[0])
    for i, j in zip(ret[0], ret_gt[0]):
        assert i == j
