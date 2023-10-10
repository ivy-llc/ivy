# global

from hypothesis import strategies as st


# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


@handle_frontend_test(
    fn_tree="tensorflow.nest.flatten",
    dtype_and_x=helpers.dtype_and_values(
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=3,
        max_dim_size=5,
    ),
    expand_composites=st.booleans(),
    use_array=st.booleans(),
)
def test_tensorflow_flatten(
    *,
    dtype_and_x,
    expand_composites,
    use_array,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        structure=x[0] if use_array else x[0].tolist(),
        expand_composites=expand_composites,
    )
