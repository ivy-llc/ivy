# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_tests.test_ivy.test_functional.test_experimental.test_core.test_linalg import (
    _generate_dot_dtype_and_arrays,
)
from ivy_tests.test_ivy.test_frontends.test_tensorflow.test_nn import (
    _generate_bias_data,
)


@handle_frontend_test(
    fn_tree="tensorflow.keras.backend.dot",
    data=_generate_dot_dtype_and_arrays(min_num_dims=2),
)
def test_tensorflow_dot(*, data, on_device, fn_tree, frontend, test_flags, backend_fw):
    (input_dtypes, x) = data
    return helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        on_device=on_device,
        frontend=frontend,
        fn_tree=fn_tree,
        rtol=0.5,
        atol=0.5,
        x=x[0],
        y=x[1],
    )


@handle_frontend_test(
    fn_tree="tensorflow.keras.backend.bias_add",
    data=_generate_bias_data(keras_backend_fn=True),
    test_with_out=st.just(False),
)
def test_tensorflow_keras_backend_bias_add(
    *,
    data,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    data_format, dtype, x, bias = data
    helpers.test_frontend_function(
        input_dtypes=dtype * 2,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        bias=bias,
        data_format=data_format,
    )
