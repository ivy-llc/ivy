import ivy
from ivy_tests.test_ivy.helpers import handle_frontend_test
import hypothesis.strategies as st
from hypothesis import given
import tensorflow

@handle_frontend_test(
    fn_tree="tensorflow.sets.intersection",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float")
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_intersection(
    *,
    dtype_and_x,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x

    # Generate the arrays using the num_arrays parameter from dtype_and_x
    x_arrays = [helpers.generate_array(dtype=input_dtype) for _ in range(dtype_and_x.num_arrays)]

    # Call the test_frontend_function with both arrays
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x_arrays[0],
        y=x_arrays[1]
    )
