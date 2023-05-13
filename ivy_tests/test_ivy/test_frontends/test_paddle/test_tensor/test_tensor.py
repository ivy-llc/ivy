# global
from hypothesis import strategies as st, assume

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_method


CLASS_TREE = "ivy.functional.frontends.paddle.Tensor"


@st.composite
def _reshape_helper(draw):
    # generate a shape s.t len(shape) > 0
    shape = draw(
        helpers.get_shape(
            allow_none=False,
            min_num_dims=1,
            max_num_dims=3,
            min_dim_size=1,
            max_dim_size=3,
        )
    )

    reshape_shape = draw(helpers.reshape_shapes(shape=shape))

    dtypes, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            shape=shape,
        )
    )
    return dtypes, x, reshape_shape


# reshape
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="reshape",
    dtype_x_shape=_reshape_helper(),
)
def test_paddle_instance_reshape(
    dtype_x_shape,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x, shape = dtype_x_shape
    assume(len(shape) != 0)
    shape = {
        "shape": shape,
    }
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


# __getitem__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="__getitem__",
    dtype_x_index=helpers.dtype_array_index(
        available_dtypes=helpers.get_dtypes("valid"),
        allow_neg_step=False,
    ),
)
def test_paddle_instance_getitem(
    dtype_x_index,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
):
    input_dtype, x, index = dtype_x_index
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtype[0]],
        init_all_as_kwargs_np={"data": x},
        method_input_dtypes=[input_dtype[1]],
        method_all_as_kwargs_np={"item": index},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )
