import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_test.test_frontends.test_numpy import helpers as np_frontend_helpers


@handle_frontend_test(
fn_tree="numpy.asmatrix",
dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(
        arr_func=[
            lambda: helpers.dtype_and_values(
                available_dtypes=helpers.get_dtypes("float"),
            )
        ],
        get_dtypes_kind="float",
    ),
number_positional_args=np_frontend_helpers.get_num_positional_args_ufunc(
    fn_name="asmatrix"
),
)
def test_numpy_asmatrix(
    dtypes_values_casting,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtypes, x, casting, dtype = dtypes_values_casting

    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0]
        out=None,
        dtype=dtype,
        casting=casting,

        )
