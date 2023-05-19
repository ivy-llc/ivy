# local
import ivy
from ivy.functional.ivy.gradients import _variable
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy.functional.frontends.jax._src.api import device_put, device_get
from hypothesis import strategies as st
from ivy.functional.frontends.jax._src.api import vmap
import jax


def _fn1(x, y):
    return ivy.matmul(x, y)


def _fn2(x, y):
    return ivy.vecdot(x, y)


def _fn3(x, y):
    return ivy.add(x, y)


# vmap
@handle_frontend_test(
    fn_tree="jax._src.api.vmap",
    func=st.sampled_from([_fn1, _fn2, _fn3]),
    dtype_and_arrays_and_axes=helpers.arrays_and_axes(
        allow_none=False,
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=2,
        max_dim_size=10,
        num=2,
        return_dtype=True,
    ),
    in_axes_as_cont=st.booleans(),
    sample_backend=st.sampled_from(["jax", "numpy", "tensorflow", "torch", "paddle"]),
)
def test_vmap(
    func,
    dtype_and_arrays_and_axes,
    in_axes_as_cont,
    sample_backend,
):
    dtype, generated_arrays, in_axes = dtype_and_arrays_and_axes
    arrays = [ivy.native_array(array) for array in generated_arrays]

    ivy.set_backend(sample_backend)

    if in_axes_as_cont:
        vmapped_func = vmap(func, in_axes=in_axes, out_axes=0)
    else:
        vmapped_func = vmap(func, in_axes=0, out_axes=0)

    assert callable(vmapped_func)

    try:
        fw_res = helpers.flatten_and_to_np(ret=vmapped_func(*arrays))
        fw_res = fw_res if len(fw_res) else None
    except Exception:
        fw_res = None

    ivy.previous_backend()

    ivy.set_backend("jax")
    arrays = [ivy.native_array(array) for array in generated_arrays]
    if in_axes_as_cont:
        jax_vmapped_func = jax.vmap(func, in_axes=in_axes, out_axes=0)
    else:
        jax_vmapped_func = jax.vmap(func, in_axes=0, out_axes=0)

    assert callable(jax_vmapped_func)

    try:
        jax_res = helpers.flatten_and_to_np(ret=jax_vmapped_func(*arrays))
        jax_res = jax_res if len(jax_res) else None
    except Exception:
        jax_res = None

    ivy.previous_backend()

    if fw_res is not None and jax_res is not None:
        helpers.value_test(
            ret_np_flat=fw_res,
            ret_np_from_gt_flat=jax_res,
            rtol=1e-1,
            atol=1e-1,
        )

    elif fw_res is None and jax_res is None:
        pass
    else:
        assert False, "One of the results is None while other isn't"


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
