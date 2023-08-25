# global
import hypothesis.strategies as st
import tensorflow as tf

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.test_functional.test_core.test_statistical import (
    _statistical_dtype_values,
)
from ivy_tests.test_ivy.helpers import handle_frontend_test


@handle_frontend_test(
    fn_tree="tf.sets.intersection",
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
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )
