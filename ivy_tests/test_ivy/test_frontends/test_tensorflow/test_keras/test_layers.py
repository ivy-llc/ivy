# global
import numpy as np
from hypothesis import strategies as st
import ivy

# local
import ivy
from hypothesis import strategies as st
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test

@handle_frontend_test(
    fn_tree="tensorflow.keras.layers.flatten",
    dtype_and_x=helpers.dtype_and_values(
    available_dtypes=helpers.get_dtypes("valid"),
    min_num_dims=3,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=4,
    ),
     test_with_out=st.just(False))
def test_tensorflow_layers_flatten(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x  = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        )