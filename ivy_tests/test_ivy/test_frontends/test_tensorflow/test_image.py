from hypothesis import strategies as st
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


@handle_frontend_test(
    fn_tree="tensorflow.image.resize", 
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ), 
    expand_composite=st.booleans(),
    use_array=st.booleans(),
    test_with_out=st.just(False),
)
def test_tensorflow_image_resize(
    *,
    dtype_and_x,
    size,
    expand_composite,
    use_array,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    size = size
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        size=size,
        fn_tree=fn_tree,
        on_device=on_device,
        structure=x[0] if use_array else x[0].tolist(),
        expand_composites=expand_composite,
    )
