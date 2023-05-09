# global
from hypothesis import strategies as st, assume

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_method


CLASS_TREE = "ivy.functional.frontends.paddle.Tensor"


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="reshape",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", full=True),
        shape=st.shared(
            helpers.get_shape(min_num_dims=2, max_num_dims=5), key="value_shape"
        ),
    ),
    shape=helpers.reshape_shapes(
        shape=st.shared(
            helpers.get_shape(min_num_dims=2, max_num_dims=5), key="value_shape"
        ),
    ),
    unpack_shape=st.booleans(),
)
def test_paddle_instance_reshape(
    dtype_x,
    shape,
    unpack_shape,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x = dtype_x
    assume(len(shape) != 0)
    shape = {
        "shape": shape,
    }
    if unpack_shape:
        method_flags.num_positional_args = len(shape["shape"]) + 1
        i = 0
        for x_ in shape["shape"]:
            shape["x{}".format(i)] = x_
            i += 1
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np=shape,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )
