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


# normalize
@handle_frontend_test(
    fn_tree="paddle.nn.functional.normalize",
    dtype_and_x_and_axis=helpers.arrays_and_axes(
        available_dtypes=helpers.get_dtypes("float"),
        num=1,
        return_dtype=True,
        force_int_axis=True,
    ),
    p=st.floats(min_value=0.1, max_value=2),
)
def test_paddle_normalize(
    *,
    dtype_and_x_and_axis,
    p,
    test_flags,
    frontend,
    on_device,
    fn_tree,
):
    dtype, x, _ = dtype_and_x_and_axis
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        on_device=on_device,
        fn_tree=fn_tree,
        x=x[0],
        p=p,
        axis=0,
    )
