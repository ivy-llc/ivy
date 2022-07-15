from hypothesis import given, strategies as st
import ivy
import ivy_tests.test_ivy.helpers as helpers
import numpy as np


def _fn1(x, y):
    return ivy.matmul(x, y)


def _fn2(x, y):
    return ivy.vecdot(x, y)


def _fn3(x, y):
    ivy.add(x, y)


@given(func=st.sampled_from([_fn1, _fn2, _fn3]),
       arrays_and_axes=helpers.arrays_and_axes(allow_none=False,
                                               min_num_dims=2,
                                               max_num_dims=5,
                                               min_dim_size=2,
                                               max_dim_size=10,
                                               num=2),
       in_axes_as_cont=st.booleans())
def test_vmap(func, arrays_and_axes, in_axes_as_cont, fw):

    generated_arrays, in_axes = arrays_and_axes
    arrays = [ivy.native_array(array) for array in generated_arrays]

    if in_axes_as_cont:
        vmapped_func = ivy.vmap(func, in_axes=in_axes, out_axes=0)
    else:
        vmapped_func = ivy.vmap(func, in_axes=0, out_axes=0)

    assert callable(vmapped_func)

    try:
        fw_res = vmapped_func(*arrays)
    except Exception:
        fw_res = None

    ivy.set_backend("jax")
    arrays = [ivy.native_array(array) for array in generated_arrays]
    if in_axes_as_cont:
        jax_vmapped_func = ivy.vmap(func, in_axes=in_axes, out_axes=0)
    else:
        jax_vmapped_func = ivy.vmap(func, in_axes=0, out_axes=0)

    assert callable(jax_vmapped_func)

    try:
        jax_res = jax_vmapped_func(*arrays)
    except Exception:
        jax_res = None

    ivy.unset_backend()

    if fw_res is not None and jax_res is not None:
        assert np.allclose(fw_res, jax_res), f"Results from {ivy.current_backend_str()} and jax are not equal"  # noqa

    elif fw_res is None and jax_res is None:
        pass
    else:
        assert False, "One of the results is None while other isn't"

    # TODO: Tune the examples generated
