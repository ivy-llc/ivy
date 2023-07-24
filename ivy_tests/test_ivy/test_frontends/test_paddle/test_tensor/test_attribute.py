# global

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


@handle_frontend_test(
    fn_tree="paddle.tensor.attribute.is_complex",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_paddle_is_complex(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, input = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=input[0],
    )


@handle_frontend_test(
    fn_tree="paddle.tensor.attribute.is_integer",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_paddle_is_integer(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    backend_fw,
    frontend,
    test_flags,
):
    input_dtype, input = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=input[0],
    )


@handle_frontend_test(
    fn_tree="paddle.tensor.attribute.is_floating_point",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_paddle_is_floating_point(
    *,
    dtype_and_x,
    on_device,
    backend_fw,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, input = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=input[0],
    )


@handle_frontend_test(
    fn_tree="paddle.tensor.attribute.real",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_paddle_real(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    backend_fw,
    frontend,
    test_flags,
):
    input_dtype, input = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=input[0],
    )


@handle_frontend_test(
    fn_tree="paddle.tensor.attribute.imag",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_paddle_imag(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    backend_fw,
    frontend,
    test_flags,
):
    input_dtype, input = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=input[0],
    )
