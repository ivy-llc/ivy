# global
from hypothesis import strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test



@handle_frontend_test(
    fn_tree="mindspore.softmax",
    dtype_x_and_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=2,
        max_axes_size=1,
        force_int_axis=True,
        valid_axis=True,
    ),
    test_with_out=st.just(False),
)


def test_mindspore_softmax(
    *,
    dtype_x_and_axis,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):   
    input_dtype, x, axis = dtype_x_and_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        axis=axis,
    )
    
