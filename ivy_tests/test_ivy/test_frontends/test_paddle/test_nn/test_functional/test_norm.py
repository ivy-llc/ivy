# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers.testing_helpers import handle_frontend_test

from ivy_tests.test_ivy.test_functional.test_nn.test_norms import (
    _generate_data_layer_norm,
)


# layer_norm
@handle_frontend_test(
    fn_tree="paddle.nn.functional.layer_norm",
    values_tuple=_generate_data_layer_norm(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    eps=st.floats(min_value=0.01, max_value=0.1),
)
def test_paddle_layer_norm(
    *,
    values_tuple,
    normalized_shape,
    eps,
    test_flags,
    frontend,
    on_device,
    backend_fw,
    fn_tree,
):
    (dtype, x, normalized_shape, scale, offset) = values_tuple
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        on_device=on_device,
        fn_tree=fn_tree,
        x=x[0],
        normalized_shape=normalized_shape,
        weight=scale[0],
        bias=offset[0],
        epsilon=eps,
    )


# instance_norm
@handle_frontend_test(
    fn_tree="paddle.nn.functional.instance_norm",
    values_tuple=_generate_data_layer_norm(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    eps=st.floats(min_value=0.01, max_value=0.1),
)
def test_paddle_instance_norm(
    *,
    eps,
    test_flags,
    values_tuple,
    frontend,
    on_device,
    fn_tree,
):
    (dtype, x, normalized_shape, scale, offset) = values_tuple
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        on_device=on_device,
        fn_tree=fn_tree,
        x=x[0],
        normalized_shape=normalized_shape,
        weight=scale[0],
        bias=offset[0],
        epsilon=eps,
    )


# testing local_response_norm
@st.composite
def _local_response_norm_helper(draw):
    dtype_and_param = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            min_num_dims=1,
            max_num_dims=6,
        )
    )

    dtype_and_indices = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            min_num_dims=1,
            max_num_dims=6,
        )
    )
    dtype, param = dtype_and_param
    dtype, indices = dtype_and_indices
    return dtype, param, indices


@handle_frontend_test(
    fn_tree="paddle.local_response_norm",
    dtype_param_and_indices=_local_response_norm_helper(),
)
def test_paddle_local_response_norm(
    *,
    dtype_param_and_indices,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, param, indices = dtype_param_and_indices
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        param=param[0],
        indices=indices[0],
    )
