# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers

from ivy_tests.test_ivy.helpers import handle_frontend_test


@st.composite
def _avg_poolnd_pad_helper(draw, min_value, max_value, num_dims=2):
    pad = draw(st.integers(min_value=min_value, max_value=max_value))
    return tuple([pad for _ in range(0, num_dims)])


# avg_pool2d
@handle_frontend_test(
    fn_tree="torch.nn.functional.avg_pool2d",
    dtype_x_k_s=helpers.arrays_for_pooling(
        min_dims=4, max_dims=4, min_side=1, max_side=4
    ),
    padding=_avg_poolnd_pad_helper(min_value=1, max_value=3, num_dims=2),
)
def test_torch_avg_pool2d(
    dtype_x_k_s,
    padding,
    *,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    input_dtype, x, kernel_size, stride, _ = dtype_x_k_s
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=None,
    )
