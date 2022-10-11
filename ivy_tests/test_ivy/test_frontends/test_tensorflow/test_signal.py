import ivy
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# hann_window
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.hann_window",
    ),
)
# hann_window
def test_hann_window(dtype_and_x, num_positional_args):
    dtype, x = dtype_and_x
    x = helpers.convert_to_tensor(x, dtype=dtype)
    if num_positional_args == 1:
        res = ivy.hann_window(x)
    elif num_positional_args == 2:
        res = ivy.hann_window(x, dtype=dtype)
    else:
        raise ValueError
    assert res.shape == x.shape
    assert res.dtype == x.dtype
    assert res.device == x.device
    assert ivy.allclose(res, ivy.numpy.hann_window(x.numpy(), dtype=dtype.numpy()))
