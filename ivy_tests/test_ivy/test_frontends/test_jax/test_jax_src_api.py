# local
import ivy
from ivy.functional.ivy.gradients import _variable
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy.functional.frontends.jax._src.api import device_put, device_get


# device_put
@handle_frontend_test(
    fn_tree="jax._src.api.device_put",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_jax_device_put(
    *,
    dtype_and_x,
    test_flags,
    fn_tree,
    frontend,
    on_device,
):
    dtype, x = dtype_and_x
    dtype = dtype[0]
    x = x[0]

    x = ivy.asarray(x)
    if test_flags.as_variable and ivy.is_float_dtype(dtype):
        x = _variable(x)

    device = ivy.dev(x)
    x_on_dev = device_put(x, on_device).ivy_array
    dev_from_new_x = ivy.dev(x_on_dev)

    # value test
    assert dev_from_new_x == device


# device_get
@handle_frontend_test(
    fn_tree="jax._src.api.device_get",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_jax_device_get(
    *,
    dtype_and_x,
    test_flags,
    fn_tree,
    frontend,
    on_device,
):
    dtype, x = dtype_and_x
    dtype = dtype[0]
    x = x[0]

    x = ivy.asarray(x)
    if test_flags.as_variable and ivy.is_float_dtype(dtype):
        x = _variable(x)

    x_on_dev = device_get(x).ivy_array
    dev_from_new_x = ivy.dev(x_on_dev)

    # value test
    assert dev_from_new_x == "cpu"
