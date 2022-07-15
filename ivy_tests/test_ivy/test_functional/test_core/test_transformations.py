from hypothesis import given, strategies as st
import hypothesis.extra.numpy as hnp
from random import choice
import numpy as np
import ivy
import ivy_tests.test_ivy.helpers as helpers
import torch

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
    except Exception as error:
        print("fw Error:", error)
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
    except Exception as error:
        print("jax Error:", error)
        jax_res = None

    ivy.clear_backend_stack()

    if fw_res is not None and jax_res is not None:
        assert ivy.array_equal(ivy.array(ivy.to_numpy(fw_res)),
                               ivy.array(ivy.to_numpy(jax_res))),\
            f"Results are not equal. fw: {fw_res}, Jax: {jax_res}"
        print(" A HIT")
        # if isinstance(in_axes, (list, tuple)):
        #     if None in in_axes:
        #         print("with a none")
        #     elif in_axes is None:
        #         print("with a none")
        # print(in_axes)

    elif fw_res is None and jax_res is None:
        pass
    else:
        # print("in_axes:", in_axes)
        # print("fw_res:", fw_res)
        # print("jax_res:", jax_res)
        assert False, "One of the results is None while other isn't"
    # TODO: Tune the examples generated

