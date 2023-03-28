import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
    


@handle_frontend_test(
fn_tree="numpy.asmatrix",
dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    get_dtypes_kind="numeric",
),
number_positional_args=np_frontend_helpers.get_num_positional_args_ufunc(
    fn_name="asmatrix"
),
)
def test_numpy_asmatrix(
    dtypes_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtypes, x = dtypes_and_x
    
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        
        out=None,
        dtype=dtype,
        
        )
